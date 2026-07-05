import json
from pathlib import Path

from build_cognitive_trace_dataset import (
    NONE_COGNITIVE_EVENT,
    NONE_HYPOTHESIS,
    build_rows,
    write_dataset,
)
from human_trace.schema import CognitiveEvent, EpisodeRecord, IntentTag, StepRecord


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.to_json() + "\n")


def test_cognitive_dataset_builds_action_phase_and_hypothesis_rows(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    game_id = "zz00-test"
    stamp = "20260615-000000"
    _write_jsonl(
        trace_dir / f"{game_id}.{stamp}.steps.jsonl",
        [
            StepRecord(
                game_id=game_id,
                episode_id="ep1",
                step=0,
                frame_before=[[0]],
                available_actions=[],
                action="RESET",
                action_args=None,
                frame_after=[[1, 0], [0, 0]],
                game_state_after="NOT_FINISHED",
                levels_completed_after=1,
                intent=IntentTag.EXPLORE_UNKNOWN.value,
                hypothesis="",
                t_ms=0,
            ),
            StepRecord(
                game_id=game_id,
                episode_id="ep1",
                step=1,
                frame_before=[[1, 0], [0, 0]],
                available_actions=[1, 6],
                action="ACTION1",
                action_args=None,
                frame_after=[[1, 1], [0, 0]],
                game_state_after="NOT_FINISHED",
                levels_completed_after=1,
                intent=IntentTag.TEST_MOVE.value,
                hypothesis="action 1 moves cursor",
                t_ms=10,
            ),
            StepRecord(
                game_id=game_id,
                episode_id="ep1",
                step=2,
                frame_before=[[1, 1], [0, 0]],
                available_actions=[1, 6],
                action="ACTION6",
                action_args={"x": 1, "y": 0},
                frame_after=[[2, 1], [0, 0]],
                game_state_after="WIN",
                levels_completed_after=2,
                intent=IntentTag.TEST_CLICK.value,
                cognitive_events=[CognitiveEvent.HYPOTHESIS_CONFIRMED.value],
                hypothesis="action 1 moves cursor",
                t_ms=20,
            ),
        ],
    )
    _write_jsonl(
        trace_dir / f"{game_id}.{stamp}.episodes.jsonl",
        [
            EpisodeRecord(
                game_id=game_id,
                episode_id="ep1",
                started_at="2026-06-15T00:00:00+00:00",
                ended_at="2026-06-15T00:01:00+00:00",
                n_steps=3,
                final_state="WIN",
                levels_completed=2,
                game_type_guess="test",
                objective_guess="click target",
                discovered_mechanics=["click paints"],
            )
        ],
    )

    rows = build_rows(trace_dir=trace_dir)

    assert [row["action"] for row in rows] == ["ACTION1", "ACTION6"]
    assert [row["cognitive_phase"] for row in rows] == ["test_move", "test_click"]
    assert rows[0]["hypothesis_label"] == "action 1 moves cursor"
    assert rows[0]["hypothesis_changed"] is True
    assert rows[1]["hypothesis_changed"] is False
    assert rows[0]["level_before"] == 1
    assert rows[1]["level_up"] is True
    assert rows[1]["click_x"] == 1
    assert rows[1]["cognitive_events"] == ["hypothesis_confirmed"]
    assert rows[1]["cognitive_event_label"] == "hypothesis_confirmed"
    assert rows[1]["hypothesis_confirmed"] is True
    assert rows[1]["hypothesis_rejected"] is False
    assert rows[1]["goal_changed"] is False
    assert rows[1]["targets"] == {
        "action": "ACTION6",
        "cognitive_phase": "test_click",
        "hypothesis": "action 1 moves cursor",
        "cognitive_event": "hypothesis_confirmed",
        "hypothesis_confirmed": True,
        "hypothesis_rejected": False,
        "goal_changed": False,
    }


def test_cognitive_dataset_can_keep_or_drop_empty_hypotheses(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    game_id = "zz01-test"
    stamp = "20260615-000001"
    _write_jsonl(
        trace_dir / f"{game_id}.{stamp}.steps.jsonl",
        [
            StepRecord(
                game_id=game_id,
                episode_id="ep1",
                step=0,
                frame_before=[[0, 0], [0, 0]],
                available_actions=[1],
                action="ACTION1",
                action_args=None,
                frame_after=[[0, 1], [0, 0]],
                game_state_after="NOT_FINISHED",
                levels_completed_after=0,
                intent=IntentTag.NONE.value,
                hypothesis="",
                t_ms=0,
            )
        ],
    )
    (trace_dir / f"{game_id}.{stamp}.episodes.jsonl").write_text("", encoding="utf-8")

    kept = build_rows(trace_dir=trace_dir)
    dropped = build_rows(trace_dir=trace_dir, include_empty_hypothesis=False)

    assert kept[0]["hypothesis_label"] == NONE_HYPOTHESIS
    assert kept[0]["cognitive_event_label"] == NONE_COGNITIVE_EVENT
    assert dropped == []


def test_cognitive_dataset_writes_schema_counts(tmp_path: Path):
    row = {
        "game_id": "zz02-test",
        "state_features": {},
        "action": "ACTION1",
        "cognitive_phase": "test_move",
        "cognitive_event_label": NONE_COGNITIVE_EVENT,
        "hypothesis_label": NONE_HYPOTHESIS,
    }
    out = tmp_path / "cognitive.jsonl"

    schema = write_dataset([row], out, trace_dir=tmp_path)

    assert out.exists()
    assert json.loads(out.with_suffix(".schema.json").read_text(encoding="utf-8"))["rows"] == 1
    assert schema["action_counts"] == {"ACTION1": 1}
    assert schema["cognitive_event_counts"] == {NONE_COGNITIVE_EVENT: 1}


def test_step_record_reads_legacy_rows_without_cognitive_events():
    row = {
        "game_id": "zz03-test",
        "episode_id": "ep1",
        "step": 0,
        "frame_before": [[0]],
        "available_actions": [1],
        "action": "ACTION1",
        "action_args": None,
        "frame_after": [[1]],
        "game_state_after": "NOT_FINISHED",
        "levels_completed_after": 0,
        "intent": IntentTag.TEST_MOVE.value,
        "hypothesis": "",
        "t_ms": 0,
    }

    record = StepRecord.from_json(json.dumps(row))

    assert record.cognitive_events == []
