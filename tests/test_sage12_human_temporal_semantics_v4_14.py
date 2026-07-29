from __future__ import annotations

import pytest

from theory.sage12.human_temporal_semantics_v4_14 import (
    HUMAN_TRAIN_GAMES,
    HORIZONS,
    ROLE_LABELS,
    TRANSFER_GAMES,
    TemporalBeliefState,
    TemporalSlotPrediction,
    TemporalTeacherRecord,
    _active_qwen_prompt,
    _blend_effect_probabilities,
    _candidate_action_plan,
    _future_targets,
    _torch_model,
    tensorize_temporal_records,
)
from theory.sage12.compiler import SLOT_EFFECTS
from theory.sage12.semantic_teacher_v4_9 import (
    SEMANTIC_EFFECTS,
    ObjectRelativeGraph,
)


def _record(
    *,
    index: int,
    episode: str = "episode",
    changed: bool = False,
) -> TemporalTeacherRecord:
    labels = {effect: False for effect in SEMANTIC_EFFECTS}
    labels["changed"] = changed
    graph = ObjectRelativeGraph(
        root={
            "action_name": "ACTION1" if index % 2 == 0 else "ACTION2",
            "action_family": "move",
            "requested_direction": "up" if index % 2 == 0 else "down",
            "root_kind": "actor",
        },
        neighbors=(
            {
                "direction": "north",
                "proximity": "near",
                "relative_size": "equal",
            },
        ),
    )
    return TemporalTeacherRecord(
        example_id=f"example-{index}",
        game_id="ar25",
        episode_id=episode,
        step=index,
        sequence_index=index,
        source_file="teacher-only.jsonl",
        pre_state_sha256=f"pre-{index}",
        post_state_sha256=f"post-{index}",
        graph=graph,
        labels=labels,
        applicable={effect: True for effect in SEMANTIC_EFFECTS},
        productive_score=float(changed),
        roles={role: False for role in ROLE_LABELS},
        horizon_progress={f"within_{horizon}": changed for horizon in HORIZONS},
        danger_within_8=False,
        steps_to_next_level=1 if changed else 128,
        distance_censored=not changed,
        discounted_progress=0.97 if changed else 0.0,
        human_teacher={"hypothesis": "teacher-only"},
    )


def test_v414_split_uses_all_human_games_and_separate_transfer_games() -> None:
    assert HUMAN_TRAIN_GAMES == (
        "ar25",
        "bp35",
        "cd82",
        "cn04",
        "dc22",
        "ft09",
    )
    assert set(HUMAN_TRAIN_GAMES).isdisjoint(TRANSFER_GAMES)
    assert len(TRANSFER_GAMES) == 8


def test_future_credit_is_temporal_not_an_immediate_win_relabel() -> None:
    rows = [
        {
            "levels_completed_after": 0,
            "game_state_after": "NOT_FINISHED",
        }
        for _ in range(10)
    ]
    rows[9] = {
        "levels_completed_after": 1,
        "game_state_after": "NOT_FINISHED",
    }
    before = [0] * 10

    progress, danger, distance, censored, discounted = _future_targets(
        rows,
        index=0,
        level_before=before,
    )

    assert not progress["within_4"]
    assert progress["within_16"]
    assert progress["within_64"]
    assert distance == 10
    assert not censored
    assert discounted == pytest.approx(0.97**10)
    assert not danger


def test_teacher_text_and_post_state_are_outside_student_view() -> None:
    record = _record(index=0)
    payload = record.to_dict()

    assert "human_annotation" not in payload["student_view"]
    assert "pre_state_sha256" not in payload["student_view"]
    assert "post_state_sha256" not in payload["student_view"]
    assert payload["teacher"]["human_annotation"]["hypothesis"]


def test_firewall_rejects_identity_in_model_graph() -> None:
    record = _record(index=0)
    graph = ObjectRelativeGraph(
        root={**record.graph.root, "game_id": "ar25"},
        neighbors=record.graph.neighbors,
    )
    with pytest.raises(ValueError, match="forbidden V4.14 student field"):
        TemporalTeacherRecord(
            **{
                **record.__dict__,
                "graph": graph,
            }
        )


def test_tensorizer_feedback_uses_only_previous_observed_effects() -> None:
    records = (
        _record(index=0, changed=True),
        _record(index=1, changed=False),
    )
    tensors = tensorize_temporal_records(
        records,
        hash_buckets=128,
        maximum_neighbors=4,
    )
    changed_index = SEMANTIC_EFFECTS.index("changed")

    assert tensors.feedback[0, changed_index] == 0.0
    assert tensors.feedback[1, changed_index] == 1.0
    assert tensors.next_mask[0, 0] == 1.0
    assert tensors.next_mask[1, 0] == 0.0


def test_temporal_model_emits_every_registered_head() -> None:
    import torch

    records = (_record(index=0), _record(index=1))
    tensors = tensorize_temporal_records(
        records,
        hash_buckets=128,
        maximum_neighbors=4,
    )
    model = _torch_model(
        hash_buckets=128,
        embedding_width=8,
        graph_hidden_width=16,
        temporal_hidden_width=24,
        identity_classes=2,
    )
    model.eval()
    with torch.inference_mode():
        output = model(
            torch.as_tensor(tensors.root_ids),
            torch.as_tensor(tensors.neighbor_ids),
            torch.as_tensor(tensors.neighbor_mask),
            torch.as_tensor(tensors.feedback),
        )

    assert output[0].shape == (2, len(SEMANTIC_EFFECTS))
    assert output[1].shape == (2, len(ROLE_LABELS))
    assert output[2].shape == (2, len(HORIZONS))
    assert output[3].shape == (2, 1)
    assert output[4].shape == (2, 1)
    assert output[5].shape == (
        2,
        len(SEMANTIC_EFFECTS) + len(ROLE_LABELS),
    )


def test_temporal_prediction_adapts_to_legacy_slot_annotation() -> None:
    prediction = TemporalSlotPrediction(
        effect_probabilities={effect: 0.25 for effect in SEMANTIC_EFFECTS},
        role_probabilities={role: 0.5 for role in ROLE_LABELS},
        progress_probabilities={f"within_{horizon}": 0.5 for horizon in HORIZONS},
        danger_within_8=0.1,
        normalized_distance=0.5,
        next_belief=TemporalBeliefState(),
    )

    annotation = prediction.as_slot_annotation("slot")

    assert annotation.slot_id == "slot"
    assert set(annotation.effect_probabilities)
    assert all(value == 0.25 for value in annotation.effect_probabilities.values())


def test_qwen_prior_blends_only_registered_slot_effects() -> None:
    temporal = {effect: 0.2 for effect in SEMANTIC_EFFECTS}
    qwen = {effect: 0.8 for effect in SLOT_EFFECTS}

    blended = _blend_effect_probabilities(temporal, qwen, weight=0.5)

    assert all(blended[effect] == pytest.approx(0.5) for effect in SLOT_EFFECTS)
    assert all(
        blended[effect] == pytest.approx(0.2)
        for effect in set(SEMANTIC_EFFECTS) - set(SLOT_EFFECTS)
    )


def test_active_action_plan_is_legal_and_padded_to_rollout_depth() -> None:
    sequence_table = {
        ("ACTION1", "ACTION2", "ACTION3"): 0.4,
        ("ACTION1", "ACTION4", "ACTION4"): 0.8,
        ("ACTION1", "ACTION6", "ACTION6"): 1.0,
    }

    plan = _candidate_action_plan(
        "ACTION1",
        ("ACTION1", "ACTION2", "ACTION3", "ACTION4"),
        sequence_table,
    )

    assert plan == ("ACTION1", "ACTION4", "ACTION4")
    assert len(plan) == 3


def test_active_qwen_prompt_is_identity_free_and_requests_14_bits() -> None:
    graph = _record(index=0).graph

    prompt = _active_qwen_prompt(
        graph,
        graph,
        recent_effects={effect: 0.0 for effect in SEMANTIC_EFFECTS},
    )

    assert "exactly 14 bits" in prompt
    assert '"game_id"' not in prompt
    assert '"row"' not in prompt
    assert '"col"' not in prompt
    assert "ar25" not in prompt
