from theory.m1.mechanic_grounded_candidates import (
    generate_mechanic_grounded_candidates,
    load_mechanic_profiles,
    run_mechanic_grounded_candidate_generation,
    summarize_candidates,
)
from theory.m1.mechanic_typing import load_observation_rows


def _row(game_id, action="ACTION1", **overrides):
    row = {
        "game_id": game_id,
        "action": action,
        "action_args": None,
        "position_changed": False,
        "object_motion_vectors": [],
        "adjacency_changed": False,
        "adjacency_pairs_before": [],
        "adjacency_pairs_after": [],
        "created_object_count": 0,
        "removed_object_count": 0,
        "object_count_changed": False,
        "object_count_delta": 0,
        "object_counts_by_color_before": {},
        "object_counts_by_color_after": {},
        "shape_changed": False,
        "object_measurements_after": [],
        "grid_shape_after": [6, 6],
        "player_moved": False,
        "changed_cell_ratio": 0.0,
    }
    row.update(overrides)
    return row


def test_generate_mechanic_grounded_candidates_from_raw_rows():
    rows = [
        _row(
            "bp35-test",
            position_changed=True,
            object_motion_vectors=[{"color": 4, "dy": -1, "dx": 0}],
            adjacency_changed=True,
            adjacency_pairs_before=[{"colors": [4, 5], "contacts": 1}],
            adjacency_pairs_after=[{"colors": [4, 5], "contacts": 3}],
            created_object_count=1,
            removed_object_count=1,
            object_count_changed=True,
            object_counts_by_color_before={"4": 1},
            object_counts_by_color_after={"4": 2},
            shape_changed=True,
            object_count_delta=1,
            object_measurements_after=[{"bbox": [0, 0, 1, 1]}],
            player_moved=True,
            changed_cell_ratio=0.2,
        )
        for _ in range(4)
    ]

    candidates = generate_mechanic_grounded_candidates(
        rows,
        mechanic_profiles={
            "bp35-test": {
                "representation_warning": (
                    "color_source_schema_misaligned_with_observed_mechanics"
                )
            }
        },
        min_support_count=2,
        min_support_rate=0.5,
    )
    candidate_types = {candidate.candidate_type for candidate in candidates}

    assert candidate_types >= {
        "object_motion_candidate",
        "contact_change_candidate",
        "object_lifecycle_candidate",
        "shape_zone_candidate",
        "position_effect_candidate",
    }
    assert all(candidate.status == "UNRESOLVED" for candidate in candidates)
    assert all(candidate.trace_support_counted_as_proof is False for candidate in candidates)
    assert all("color_source" not in candidate.candidate_type for candidate in candidates)
    motion = next(
        candidate
        for candidate in candidates
        if candidate.candidate_type == "object_motion_candidate"
    )
    assert motion.evidence["dominant_motion_colors"][0]["color"] == 4
    assert motion.evidence["mechanic_typing_warning"].startswith("color_source")


def test_summarize_candidates_reports_counts_by_game_and_type():
    candidates = generate_mechanic_grounded_candidates(
        [
            _row(
                "aa-game",
                position_changed=True,
                object_motion_vectors=[{"color": 1, "dy": 1, "dx": 0}],
            )
            for _ in range(3)
        ],
        min_support_count=2,
        min_support_rate=0.5,
    )
    summary = summarize_candidates(candidates)

    assert summary["candidate_count_by_game"]["aa-game"] >= 1
    assert "object_motion_candidate" in summary["candidate_count_by_type"]
    assert "object_motion_candidate" in summary["non_color_candidate_types"]


def test_real_mechanic_grounded_candidates_cover_bp35_cd82_without_color_source():
    rows = load_observation_rows("training/m1_observation_dataset.jsonl")
    profiles = load_mechanic_profiles("diagnostics/m1/mechanic_typing.json")
    candidates = generate_mechanic_grounded_candidates(
        rows,
        mechanic_profiles=profiles,
        min_support_count=3,
        min_support_rate=0.35,
        max_candidates_per_game=12,
    )
    by_game = {}
    for candidate in candidates:
        by_game.setdefault(candidate.game_id, set()).add(candidate.candidate_type)

    assert "object_motion_candidate" in by_game["bp35-0a0ad940"]
    assert "object_lifecycle_candidate" in by_game["bp35-0a0ad940"]
    assert "object_lifecycle_candidate" in by_game["cd82-fb555c5d"]
    assert "contact_change_candidate" in by_game["cd82-fb555c5d"]
    assert all("color_source" not in candidate.candidate_type for candidate in candidates)


def test_run_mechanic_grounded_candidate_generation_payload_is_unresolved():
    payload = run_mechanic_grounded_candidate_generation(
        observation_path="training/m1_observation_dataset.jsonl",
        mechanic_typing_path="diagnostics/m1/mechanic_typing.json",
        min_support_count=3,
        min_support_rate=0.35,
        max_candidates_per_game=4,
    )

    assert payload["status"] == "UNRESOLVED"
    assert payload["trace_support_counted_as_proof"] is False
    assert payload["prior_counted_as_proof"] is False
    assert payload["summary"]["candidate_count"] > 0
