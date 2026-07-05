import json
from pathlib import Path

from theory.m1.observation_dataset import (
    RawTransitionObservation,
    build_dataset,
    load_observations_jsonl,
)


THEORY_PREDICATE_NAMES = {
    "source_target_color_transform",
    "paired_with",
    "same_shape",
    "aligned_with",
    "adjacent_to",
}


def _write_trace(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _flatten_keys(value, prefix=""):
    keys = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_flatten_keys(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_flatten_keys(child, prefix))
    return keys


def test_m1_observation_dataset_round_trips_jsonl(tmp_path: Path):
    trace = tmp_path / "zz00-test.steps.jsonl"
    out = tmp_path / "m1_observations.jsonl"
    _write_trace(
        trace,
        [
            {
                "game_id": "zz00-test",
                "episode_id": "ep1",
                "step": 0,
                "frame_before": [[0]],
                "available_actions": [],
                "action": "RESET",
                "action_args": None,
                "frame_after": [[1, 0], [0, 0]],
                "game_state_after": "NOT_FINISHED",
                "levels_completed_after": 0,
                "intent": "explore_unknown",
                "hypothesis": "",
                "t_ms": 0,
            },
            {
                "game_id": "zz00-test",
                "episode_id": "ep1",
                "step": 1,
                "frame_before": [[1, 0], [0, 0]],
                "available_actions": [1],
                "action": 1,
                "action_args": {"x": 1, "y": 0},
                "frame_after": [[0, 1], [0, 0]],
                "game_state_after": "WIN",
                "levels_completed_after": 1,
                "intent": "test_move",
                "hypothesis": "action moves a small cell",
                "t_ms": 10,
            },
        ],
    )

    observations = build_dataset([trace], output_path=out)
    loaded = load_observations_jsonl(out)

    assert len(observations) == 2
    assert len(loaded) == 2
    assert loaded[1].game_id == "zz00-test"
    assert loaded[1].episode_id == "ep1"
    assert loaded[1].action == "ACTION1"
    assert loaded[1].action_args == {"x": 1, "y": 0}
    assert loaded[1].num_cells_changed == 2
    assert loaded[1].position_changed is True
    assert loaded[1].player_moved is True
    assert loaded[1].level_progressed is True
    assert loaded[1].trace_support_counted_as_proof is False
    assert loaded[1].prior_counted_as_proof is False
    assert RawTransitionObservation.from_json(loaded[1].to_json()).to_dict() == loaded[1].to_dict()


def test_m1_observation_dataset_has_game_id_and_no_theory_predicate_fields(tmp_path: Path):
    trace = tmp_path / "zz01-test.steps.jsonl"
    out = tmp_path / "m1_observations.jsonl"
    _write_trace(
        trace,
        [
            {
                "game_id": "zz01-test",
                "episode_id": "ep1",
                "step": 0,
                "frame_before": [[2, 0], [0, 3]],
                "available_actions": ["ACTION6"],
                "action": "ACTION6",
                "action_args": None,
                "frame_after": [[2, 3], [0, 0]],
                "game_state_after": "NOT_FINISHED",
                "levels_completed_after": 0,
                "intent": "test_click",
                "hypothesis": "",
                "t_ms": 5,
            }
        ],
    )

    observations = build_dataset([trace], output_path=out)
    row = observations[0].to_dict()

    assert row["game_id"] == "zz01-test"
    assert row["shape_changed"] in {True, False}
    assert row["color_changed"] is True
    assert row["adjacency_changed"] in {True, False}
    assert row["object_measurements_before"]
    assert row["counts_by_color_before"]
    assert THEORY_PREDICATE_NAMES.isdisjoint(set(_flatten_keys(row)))


def test_m1_observation_dataset_can_build_real_trace_sample(tmp_path: Path):
    trace_paths = sorted(Path("human_traces").glob("*.steps.jsonl"))
    out = tmp_path / "m1_real_sample.jsonl"

    observations = build_dataset(trace_paths, output_path=out, max_observations=1300)

    assert len(observations) >= 800
    assert len({observation.game_id for observation in observations}) >= 2
    assert out.exists()
    assert all(observation.game_id for observation in observations)
    assert all(
        observation.trace_support_counted_as_proof is False
        and observation.prior_counted_as_proof is False
        for observation in observations
    )
