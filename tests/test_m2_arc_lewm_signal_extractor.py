import json

import pytest
import torch

from theory.m2.arc_lewm_model import ARCLeWMModel
from theory.m2.arc_lewm_signal_extractor import (
    CANDIDATE_SIGNAL_FAMILIES,
    run_arc_lewm_signal_extractor,
)


def _write_dataset(path):
    games = [
        "ar25-e3c63847",
        "bp35-0a0ad940",
        "cd82-fb555c5d",
        "cn04-65d47d14",
        "dc22-4c9bff3e",
        "ft09-0d8bbf25",
    ]
    rows = []
    for index, game_id in enumerate(games, start=1):
        rows.append(
            {
                "game_id": game_id,
                "episode_id": f"{game_id}::episode",
                "step": 0,
                "grid_t": [[index, 0], [0, index]],
                "grid_t1": [[index, index], [0, index]],
                "action": "ACTION6",
                "action_args": {"x": index, "y": index + 1},
                "available_actions_t": ["ACTION3", "ACTION4", "ACTION6"],
                "terminal_t1": False,
                "level_delta": 0,
                "hud": {"available": False},
                "candidate_only_metadata": {"support": 0},
            }
        )
        rows.append(
            {
                "game_id": game_id,
                "episode_id": f"{game_id}::episode",
                "step": 1,
                "grid_t": [[index, index], [0, index]],
                "grid_t1": [[index, index], [index, index]],
                "action": "ACTION4",
                "action_args": {},
                "available_actions_t": ["ACTION3", "ACTION4", "ACTION6"],
                "terminal_t1": index == len(games),
                "level_delta": 0,
                "hud": {"available": False},
                "candidate_only_metadata": {"support": 0},
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_packet(path):
    path.write_text(
        json.dumps(
            {
                "mechanistic_context_candidates": [
                    {
                        "candidate_id": "ctx::progress",
                        "candidate_type": "relation_progress_context",
                        "support": 0,
                        "truth_status": "NOT_EVALUATED_BY_M2",
                    }
                ],
                "summary": {
                    "support": 0,
                    "truth_status": "NOT_EVALUATED_BY_M2",
                    "hud_counted_as_objective_signal": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_checkpoint(path):
    model = ARCLeWMModel(latent_dim=16, use_hud=False)
    torch.save(
        {
            "schema_version": "m2.arc_lewm_model_state.v1",
            "state_dict": model.state_dict(),
            "config": {
                "latent_dim": 16,
                "use_hud": False,
                "canvas_size": [8, 8],
            },
            "candidate_only_metadata": {
                "support": 0,
                "truth_status": "NOT_EVALUATED_BY_M2",
            },
        },
        path,
    )


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_arc_lewm_signal_extractor_builds_candidate_only_report(tmp_path):
    dataset = tmp_path / "transitions.jsonl"
    packet = tmp_path / "packet.json"
    checkpoint = tmp_path / "m2_arc_lewm.pt"
    out = tmp_path / "signal_report.json"
    _write_dataset(dataset)
    _write_packet(packet)
    _write_checkpoint(checkpoint)

    report = run_arc_lewm_signal_extractor(
        model_path=checkpoint,
        dataset_path=dataset,
        semantic_packet_path=packet,
        output_path=out,
        top_k=3,
        batch_size=4,
        device="cpu",
    )

    assert out.exists()
    assert report["schema_version"] == "m2.arc_lewm_signal_report.v1"
    assert report["candidate_signal_families"] == list(CANDIDATE_SIGNAL_FAMILIES)
    assert report["latent_prediction_quality"]["prediction_loss_val"] >= 0
    assert report["signals"]["high_surprise_transitions"]
    high_surprise = report["signals"]["high_surprise_transitions"][0]
    assert high_surprise["source_transition_id"].startswith("m2_14d::")
    assert high_surprise["context_state_origin"] == "human_trace_frame_before"
    assert high_surprise["replayability"] == "OFFLINE_TRACE_CONTEXT_ONLY"
    assert report["signals"]["low_surprise_stable_transitions"]
    assert report["signals"]["action_conditioned_delta_clusters"]
    assert report["signals"]["terminal_like_latent_neighborhoods"][
        "terminal_transition_count"
    ] == 1
    assert report["signals"]["proxy_completion_gap_candidates"]
    assert report["semantic_context_summary"]["semantic_packet_loaded"] is True

    guarded = [row for row in _walk(report) if "support" in row]
    assert guarded
    for row in guarded:
        assert row["support"] == 0
        if "world_model_score_counted_as_support" in row:
            assert row["world_model_score_counted_as_support"] is False
    assert report["contract"]["world_model_counted_as_evidence"] is False
    assert report["contract"]["a32_write_performed"] is False
    assert report["contract"]["a33_write_performed"] is False


def test_arc_lewm_signal_extractor_requires_trained_model(tmp_path):
    dataset = tmp_path / "transitions.jsonl"
    packet = tmp_path / "packet.json"
    _write_dataset(dataset)
    _write_packet(packet)

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        run_arc_lewm_signal_extractor(
            model_path=tmp_path / "missing.pt",
            dataset_path=dataset,
            semantic_packet_path=packet,
            device="cpu",
        )
