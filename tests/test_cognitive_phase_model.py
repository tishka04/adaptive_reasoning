import json
from pathlib import Path

from build_cognitive_trace_dataset import PHASE_LABELS
from cognitive_taxonomy import TAXONOMY_V2_LABELS, map_cognitive_phase
from train_cognitive_phase_model import train_and_evaluate


def _row(index: int, *, phase: str, episode: str):
    value = float(index)
    return {
        "game_id": "toy-game",
        "episode_id": episode,
        "state_features": {
            "component_count": value,
            "distinct_colors": value % 3,
        },
        "history_features": {},
        "cognitive_phase": phase,
        "hypothesis_label": "__none__",
    }


def test_cognitive_phase_training_outputs_cv_metrics_and_model(tmp_path: Path):
    dataset = tmp_path / "cognitive.jsonl"
    rows = []
    labels = PHASE_LABELS[:3]
    for group_index in range(9):
        phase = labels[group_index % len(labels)]
        for j in range(6):
            rows.append(_row(group_index * 10 + j, phase=phase, episode=f"ep{group_index}"))
    with dataset.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    model_out = tmp_path / "cognitive_phase.joblib"
    metrics = train_and_evaluate(
        dataset_path=str(dataset),
        model_out=str(model_out),
        games="all",
        n_splits=3,
        seed=0,
        n_estimators=20,
        min_samples_leaf=1,
        quiet=True,
    )

    assert model_out.exists()
    assert model_out.with_suffix(".metrics.json").exists()
    assert metrics["rows"] == len(rows)
    assert metrics["n_splits"] == 3
    assert "macro_f1" in metrics["oof"]
    assert set(metrics["per_class"]) == set(PHASE_LABELS)


def test_taxonomy_v2_splits_meta_classes_and_drops_none():
    assert (
        map_cognitive_phase(
            {
                "cognitive_phase": "explore_unknown",
                "action": "ACTION6",
                "click_x": 3,
                "hypothesis_label": "click_based_game",
            },
            "v2",
        )
        == "explore_click"
    )
    assert (
        map_cognitive_phase(
            {
                "cognitive_phase": "probe_object",
                "action": "ACTION6",
                "hypothesis_label": "use_bridges_to_reach_target",
            },
            "v2",
        )
        == "probe_bridge_or_path"
    )
    assert map_cognitive_phase({"cognitive_phase": "none"}, "v2") is None


def test_cognitive_phase_training_can_use_taxonomy_v2(tmp_path: Path):
    dataset = tmp_path / "cognitive_v2.jsonl"
    rows = []
    phases = [
        ("explore_unknown", "ACTION1", "action1_moves_cursor"),
        ("explore_unknown", "ACTION6", "click_based_game"),
        ("probe_object", "ACTION6", "match_centered_shapes_with_colors"),
        ("test_click", "ACTION6", "click_on_button"),
        ("test_interaction", "ACTION2", "shape_changes_color"),
        ("repeat_success", "ACTION1", "known_solution"),
        ("none", "ACTION1", ""),
    ]
    for group_index in range(14):
        phase, action, hypothesis = phases[group_index % len(phases)]
        for j in range(4):
            row = _row(group_index * 10 + j, phase=phase, episode=f"ep{group_index}")
            row["action"] = action
            row["hypothesis_label"] = hypothesis or "__none__"
            row["click_x"] = 1 if action == "ACTION6" else None
            rows.append(row)
    with dataset.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    model_out = tmp_path / "cognitive_phase_v2.joblib"
    metrics = train_and_evaluate(
        dataset_path=str(dataset),
        model_out=str(model_out),
        games="all",
        n_splits=2,
        seed=0,
        n_estimators=10,
        min_samples_leaf=1,
        taxonomy="v2",
        quiet=True,
    )

    assert model_out.exists()
    assert metrics["taxonomy"] == "v2"
    assert "none" not in metrics["per_class"]
    assert set(metrics["per_class"]) == set(TAXONOMY_V2_LABELS)
    assert metrics["label_counts"]["test_control_or_activation"] > 0
    assert metrics["label_counts"]["test_object_interaction"] > 0
