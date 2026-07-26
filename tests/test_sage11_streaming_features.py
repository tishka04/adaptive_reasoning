"""Parity and lifecycle tests for the shared SAGE.11 v2 feature interface."""

from __future__ import annotations

import numpy as np

from theory.sage11.streaming_features import (
    STREAMING_FEATURE_FORMAT_VERSION,
    StreamingFeatureSchema,
    StreamingFeatureTracker,
    encode_transition_rows,
)


def _effects(changed: str, moved: str) -> list[str]:
    return [
        f"effect:changed_cells({changed})",
        f"effect:player_moved({moved})",
        (
            f"effect:value_multiset_delta({changed},{changed})"
            if changed != "zero"
            else ""
        ),
        "progress:level_complete(False)",
        "risk:game_over(False)",
    ]


def _rows() -> list[dict]:
    first_effects = [atom for atom in _effects("few", "True") if atom]
    second_effects = [atom for atom in _effects("zero", "False") if atom]
    return [
        {
            "game_id": "g000",
            "seed": 0,
            "reset_index": 0,
            "step_index": 0,
            "source_split": "source_train",
            "action_name": "ACTION6",
            "action_data": {"x": 0, "y": 0},
            "atoms_before": [
                "action:available(ACTION1)",
                "object:role_present(one:square)",
            ],
            "atoms_after": [
                "action:available(ACTION1)",
                "object:role_present(few:square)",
            ],
            "state_digest_before": "state-a",
            "state_digest_after": "state-b",
            "effect_atoms": first_effects,
        },
        {
            "game_id": "g000",
            "seed": 0,
            "reset_index": 0,
            "step_index": 1,
            "source_split": "source_train",
            "action_name": "ACTION6",
            "action_data": {"x": 3, "y": 4},
            "atoms_before": [
                "action:available(ACTION1)",
                "object:role_present(few:square)",
            ],
            "atoms_after": [
                "action:available(ACTION1)",
                "object:role_present(few:square)",
            ],
            "state_digest_before": "state-b",
            "state_digest_after": "state-b",
            "effect_atoms": second_effects,
        },
    ]


def test_shared_row_loader_matches_direct_streaming_lifecycle():
    rows = _rows()
    dataset = encode_transition_rows(
        lambda: iter(rows),
        total_rows=len(rows),
        manifest_checksum="manifest",
    )
    tracker = StreamingFeatureTracker(dataset.schema)
    direct = []
    for row in rows:
        context = tracker.begin_step(
            sequence_key=(
                row["game_id"],
                row["seed"],
                row["reset_index"],
            ),
            step_index=row["step_index"],
            atoms_before=row["atoms_before"],
            state_digest_before=row["state_digest_before"],
        )
        direct.append(tracker.encode_action(
            context,
            action_name=row["action_name"],
            action_data=row["action_data"],
        ))
        tracker.observe_transition(
            context,
            action_name=row["action_name"],
            action_data=row["action_data"],
            atoms_after=row["atoms_after"],
            state_digest_after=row["state_digest_after"],
            effect_atoms=row["effect_atoms"],
        )
    assert np.array_equal(dataset.features, np.stack(direct))
    assert dataset.exact_continuity.tolist() == [False, True]
    assert dataset.labels["changed_cells"].tolist() == [2, 0]
    assert dataset.labels["player_moved"].tolist() == [1, 0]


def test_candidate_encoding_does_not_advance_streaming_state():
    schema = StreamingFeatureSchema.fit([(
        "action:available(ACTION1)",
        "object:role_present(one:square)",
    )])
    tracker = StreamingFeatureTracker(schema)
    context = tracker.begin_step(
        sequence_key="live",
        step_index=0,
        atoms_before=schema.atom_vocabulary,
        state_digest_before="same",
    )
    action1 = tracker.encode_action(
        context,
        action_name="ACTION1",
    )
    action2 = tracker.encode_action(
        context,
        action_name="ACTION2",
    )
    mapping = schema.feature_to_index
    assert action1[mapping["state_visit:first"]] == 1.0
    assert action2[mapping["state_visit:first"]] == 1.0
    assert action1[mapping["current_action:ACTION1"]] == 1.0
    assert action2[mapping["current_action:ACTION2"]] == 1.0


def test_streaming_schema_round_trip_and_shortcut_partitions():
    schema = StreamingFeatureSchema.fit([(
        "action:available(ACTION1)",
        "object:role_present(one:square)",
        "state:game_state(not_finished)",
    )])
    restored = StreamingFeatureSchema.from_dict(schema.to_dict())
    assert restored == schema
    assert restored.format_version == STREAMING_FEATURE_FORMAT_VERSION
    assert set(restored.action_feature_indices).isdisjoint(
        restored.state_only_feature_indices
    )
    signature_names = {
        restored.feature_names[index]
        for index in restored.game_signature_feature_indices
    }
    assert signature_names == {
        "current_atom:action:available(ACTION1)",
        "current_atom:object:role_present(one:square)",
    }
