import json

from theory.m2.arc_lewm_dataset import (
    build_source_audit,
    run_arc_lewm_dataset_builder,
)


def _write_steps(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(
    *,
    game_id="bp35-0a0ad940",
    episode_id="run_001",
    step=0,
    frame_before=None,
    frame_after=None,
    action="ACTION3",
    action_args=None,
    available_actions=None,
    game_state_after="NOT_FINISHED",
):
    row = {
        "game_id": game_id,
        "episode_id": episode_id,
        "step": step,
        "frame_before": frame_before if frame_before is not None else [[1]],
        "available_actions": available_actions if available_actions is not None else [3, 4, 6],
        "action": action,
        "action_args": action_args,
        "frame_after": frame_after if frame_after is not None else [[1]],
        "game_state_after": game_state_after,
        "levels_completed_after": 0,
    }
    if frame_after is None:
        row["frame_after"] = [[1]]
    return row


def test_arc_lewm_dataset_builds_row_local_transitions_and_split(tmp_path):
    traces = tmp_path / "human_traces"
    traces.mkdir()
    _write_steps(
        traces / "bp35-0a0ad940.steps.jsonl",
        [
            _row(step=0, frame_before=[[0]], frame_after=[[1]], action="RESET", available_actions=[]),
            _row(
                step=1,
                frame_before=[[1]],
                frame_after=[[2]],
                action="ACTION6",
                action_args={"x": 30, "y": 12},
            ),
            _row(
                step=2,
                frame_before=[[2]],
                frame_after=[[2]],
                action="ACTION3",
                game_state_after="GAME_OVER",
            ),
        ],
    )
    _write_steps(
        traces / "ar25-e3c63847.steps.jsonl",
        [
            _row(
                game_id="ar25-e3c63847",
                step=0,
                frame_before=[[0]],
                frame_after=[[5]],
                action="RESET",
                available_actions=[],
            ),
            _row(
                game_id="ar25-e3c63847",
                step=1,
                frame_before=[[5]],
                frame_after=[[6]],
                action=1,
                available_actions=[1, 2, 3],
            ),
        ],
    )

    payload = run_arc_lewm_dataset_builder(traces_dir=traces)

    transitions = payload["transitions"]
    action6 = [row for row in transitions if row["action"] == "ACTION6"][0]
    assert action6["grid_t"] == [[1]]
    assert action6["grid_t1"] == [[2]]
    assert action6["action_args"] == {"x": 30, "y": 12}
    assert action6["available_actions_t"] == ["ACTION3", "ACTION4", "ACTION6"]
    assert action6["hud"] == {
        "available": False,
        "source": "not_extracted_in_m2_14_foundation",
    }
    assert action6["candidate_only_metadata"]["support"] == 0
    assert action6["candidate_only_metadata"]["truth_status"] == "NOT_EVALUATED_BY_M2"
    assert action6["candidate_only_metadata"]["trace_support_counted_as_proof"] is False

    terminal = [row for row in transitions if row["terminal_t1"]]
    assert terminal
    manifest = payload["manifest"]
    assert manifest["summary"]["games_total"] == 2
    assert manifest["split"]["mode"] == "by_game"
    assert len(manifest["split"]["train_games"]) == 1
    assert len(manifest["split"]["val_games"]) == 1
    assert manifest["alignment_policy"]["transition_alignment_policy"] == (
        "row_local_frame_before_action_frame_after"
    )
    assert manifest["alignment_policy"]["off_by_one_pairing_used"] is False
    assert manifest["continuity_checks"]["skipped_reset_boundaries"] >= 2
    assert manifest["summary"]["support"] == 0
    assert manifest["summary"]["a32_write_performed"] is False
    assert manifest["summary"]["a33_write_performed"] is False


def test_arc_lewm_dataset_rejects_rows_missing_grid_t1(tmp_path):
    traces = tmp_path / "human_traces"
    traces.mkdir()
    invalid = _row(step=0, frame_before=[[1]], frame_after=[[2]])
    invalid.pop("frame_after")
    _write_steps(traces / "bp35-0a0ad940.steps.jsonl", [invalid])

    payload = run_arc_lewm_dataset_builder(traces_dir=traces)

    assert payload["transitions"] == []
    assert payload["manifest"]["summary"]["rows_rejected_missing_grid_t1"] == 1
    assert payload["manifest"]["rejected_rows"][0]["support"] == 0


def test_arc_lewm_dataset_reports_continuity_mismatch_without_repairing(tmp_path):
    traces = tmp_path / "human_traces"
    traces.mkdir()
    _write_steps(
        traces / "bp35-0a0ad940.steps.jsonl",
        [
            _row(step=1, frame_before=[[1]], frame_after=[[9]], action="ACTION3"),
            _row(step=2, frame_before=[[2]], frame_after=[[3]], action="ACTION4"),
        ],
    )

    payload = run_arc_lewm_dataset_builder(traces_dir=traces)

    assert payload["manifest"]["alignment_policy"]["continuity_mismatches"] == 1
    second = [row for row in payload["transitions"] if row["step"] == 2][0]
    assert second["grid_t"] == [[2]]
    assert second["grid_t1"] == [[3]]


def test_arc_lewm_source_audit_rejects_unverified_derived_logs():
    audit = build_source_audit()

    assert audit["accepted_source_types"] == ["human_traces_steps_jsonl"]
    assert "m1_observation_dataset_jsonl" in audit["rejected_source_types"]
    assert "m3_experiment_logs" in audit["rejected_source_types"]
    assert audit["silent_ingestion_performed"] is False
    assert audit["support"] == 0
    assert audit["truth_status"] == "NOT_EVALUATED_BY_M2"
