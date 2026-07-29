from __future__ import annotations

import numpy as np
import pytest

from theory.live_transition_loop import build_observation
from theory.sage12.action_target_data import resolve_action_target
from theory.sage12.demonstration_milestone_policy_v4_15 import (
    MILESTONE_LABELS,
    DemonstrationChoiceRecord,
    _candidate_graph,
    _candidate_graph_from_observation,
    _choice_batch,
    _next_milestone,
    _suffix_return,
    _torch_model,
    _zscore,
    tensorize_choices,
)
from theory.sage12.semantic_teacher_v4_9 import (
    SEMANTIC_EFFECTS,
    ObjectRelativeGraph,
)


def _graph(action: str, *, direction: str) -> ObjectRelativeGraph:
    return ObjectRelativeGraph(
        root={
            "action_name": action,
            "action_family": "move",
            "requested_direction": direction,
            "root_kind": "actor",
            "root_occupied": 0,
            "path_status": "open",
        },
        neighbors=(
            {
                "direction": direction,
                "proximity": "near",
                "relative_size": "equal",
                "is_actor": 0,
            },
        ),
    )


def _record(index: int) -> DemonstrationChoiceRecord:
    candidates = (
        _graph("ACTION1", direction="north"),
        _graph("ACTION2", direction="south"),
    )
    effects = {effect: False for effect in SEMANTIC_EFFECTS}
    effects["productive"] = index == 0
    return DemonstrationChoiceRecord(
        example_id=f"example-{index}",
        game_id="ar25",
        episode_id="episode",
        sequence_index=index,
        source_file="teacher.jsonl",
        pre_state_sha256=f"pre-{index}",
        post_state_sha256=f"post-{index}",
        candidates=candidates,
        selected_index=index % 2,
        observed_effects=effects,
        milestone="productive" if index == 0 else "none_within_64",
        milestone_distance=1 if index == 0 else 64,
        suffix_return=0.8 if index == 0 else 0.0,
        success_weight=4.2 if index == 0 else 1.0,
        within_16=index == 0,
        productive=index == 0,
        actual_action_name="ACTION1" if index == 0 else "ACTION2",
        actual_action_data={},
    )


def test_student_view_excludes_audit_identity_and_coordinates() -> None:
    payload = _record(0).to_dict()

    assert "game_id" not in payload["student_view"]
    assert "pre_state_sha256" not in payload["student_view"]
    assert "actual_action_data" not in payload["student_view"]
    assert payload["audit"]["game_id"] == "ar25"


def test_firewall_rejects_identity_inside_candidate_graph() -> None:
    record = _record(0)
    unsafe = ObjectRelativeGraph(
        root={**record.candidates[0].root, "game_id": "ar25"},
        neighbors=record.candidates[0].neighbors,
    )

    with pytest.raises(ValueError, match="forbidden V4.15 student field"):
        DemonstrationChoiceRecord(
            **{
                **record.__dict__,
                "candidates": (unsafe, record.candidates[1]),
            }
        )


def test_milestone_teacher_selects_earliest_prioritized_event() -> None:
    rows = []
    for _ in range(5):
        rows.append({"_labels": {effect: False for effect in SEMANTIC_EFFECTS}})
    rows[2]["_labels"]["target_removed"] = True
    rows[2]["_labels"]["productive"] = True
    rows[4]["_labels"]["level_complete"] = True

    milestone, distance = _next_milestone(rows, index=0)

    assert milestone == "target_removed"
    assert distance == 3


def test_suffix_return_rewards_progress_and_penalizes_danger() -> None:
    rows = [
        {"_productive_score": 1.0},
        {"_productive_score": 1.0},
    ]

    safe = _suffix_return(
        rows,
        index=0,
        discounted_progress=0.8,
        danger=False,
    )
    dangerous = _suffix_return(
        rows,
        index=0,
        discounted_progress=0.8,
        danger=True,
    )

    assert safe > dangerous
    assert -1.0 <= dangerous <= safe <= 1.0


def test_tensorizer_preserves_candidate_mask_and_sequence() -> None:
    tensors = tensorize_choices(
        (_record(0), _record(1)),
        hash_buckets=128,
        maximum_neighbors=4,
    )

    assert tensors.candidate_mask.shape == (2, 2)
    assert tensors.candidate_mask.sum() == 4
    assert tensors.selected_index.tolist() == [0, 1]
    assert tensors.sequences == ((0, 1),)


def test_policy_context_is_causal_and_heads_cover_registered_targets() -> None:
    import torch

    records = (_record(0), _record(1))
    tensors = tensorize_choices(
        records,
        hash_buckets=128,
        maximum_neighbors=4,
    )
    batch, _mask = _choice_batch(
        tensors,
        ((0, 1),),
        device="cpu",
    )
    model = _torch_model(
        hash_buckets=128,
        embedding_width=8,
        graph_hidden_width=16,
        temporal_hidden_width=24,
        milestone_embedding_width=6,
    )
    model.eval()
    with torch.inference_mode():
        output = model(
            batch["candidate_root_ids"],
            batch["candidate_neighbor_ids"],
            batch["candidate_neighbor_mask"],
            batch["candidate_mask"],
            batch["selected_index"],
            batch["observed_effects"],
        )

    assert torch.count_nonzero(output["contexts"][0, 0]) == 0
    assert output["contexts"].shape == (1, 2, 24)
    assert output["candidate_latent"].shape == (1, 2, 2, 16)
    assert output["milestone_logits"].shape == (
        1,
        2,
        len(MILESTONE_LABELS),
    )
    assert output["distance"].shape == (1, 2)
    assert output["suffix_return"].shape == (1, 2)


def test_zscore_is_stable_for_ties_and_centered_otherwise() -> None:
    assert np.allclose(_zscore((1.0, 1.0)), 0.0)
    transformed = _zscore((1.0, 2.0, 3.0))
    assert float(transformed.mean()) == pytest.approx(0.0)
    assert float(transformed.std()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("action_name", "action_data"),
    (
        ("ACTION1", {}),
        ("ACTION6", {"x": 3, "y": 1}),
        ("ACTION5", {}),
    ),
)
def test_cached_candidate_graph_is_exactly_equivalent(
    action_name: str,
    action_data: dict[str, int],
) -> None:
    frame = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, 2, 0, 3, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 4, 4, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    available = ("ACTION1", "ACTION5", "ACTION6")
    observation = build_observation(
        frame,
        available_actions=available,
        game_state="NOT_FINISHED",
        levels_completed=0,
        infer_players=True,
    )
    anchor = resolve_action_target(observation, action_name, action_data)

    cached = _candidate_graph_from_observation(
        game="ar25",
        observation=observation,
        anchor=anchor,
        action_name=action_name,
    )
    legacy = _candidate_graph(
        game="ar25",
        sequence_index=0,
        available=available,
        action_name=action_name,
        action_data=action_data,
        frame=frame,
        game_state="NOT_FINISHED",
        levels_completed=0,
    )

    assert cached == legacy
