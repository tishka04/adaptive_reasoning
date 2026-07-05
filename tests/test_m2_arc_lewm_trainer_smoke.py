import json

from theory.m2.arc_lewm_trainer import train_arc_lewm


def _transition(game_id, step, before, after, action="ACTION3", action_args=None):
    return {
        "game_id": game_id,
        "episode_id": f"{game_id}::episode",
        "step": step,
        "grid_t": before,
        "grid_t1": after,
        "action": action,
        "action_args": action_args or {},
        "available_actions_t": ["ACTION3", "ACTION4", "ACTION6"],
        "terminal_t1": False,
        "level_delta": 0,
        "hud": {
            "available": False,
            "source": "not_extracted_in_m2_14_foundation",
        },
        "candidate_only_metadata": {
            "support": 0,
            "truth_status": "NOT_EVALUATED_BY_M2",
            "trace_support_counted_as_proof": False,
        },
    }


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
            _transition(
                game_id,
                0,
                [[index, 0], [0, index]],
                [[index, index], [0, index]],
                action="ACTION6",
                action_args={"x": index + 1, "y": index},
            )
        )
        rows.append(
            _transition(
                game_id,
                1,
                [[index, index], [0, index]],
                [[index, index], [index, index]],
                action="ACTION4",
            )
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_arc_lewm_trainer_smoke_completes_without_collapse(tmp_path):
    dataset = tmp_path / "transitions.jsonl"
    model_path = tmp_path / "m2_arc_lewm.pt"
    report_path = tmp_path / "report.json"
    _write_dataset(dataset)

    report = train_arc_lewm(
        dataset_path=dataset,
        output_model_path=model_path,
        report_path=report_path,
        latent_dim=16,
        lambda_sigreg=0.1,
        use_hud=False,
        split="by_game",
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        canvas_size=(8, 8),
        device="cpu",
    )

    assert model_path.exists()
    assert (tmp_path / "m2_arc_lewm_best_val.pt").exists()
    assert (tmp_path / "m2_arc_lewm_last.pt").exists()
    assert report_path.exists()
    summary = report["summary"]
    diagnostics = report["anti_collapse_diagnostics"]
    baselines = report["baseline_diagnostics"]
    assert summary["training_completed"] is True
    assert summary["training_steps_completed"] is True
    assert summary["prediction_loss_decreased_or_stable"] is True
    assert summary["sigreg_active"] is True
    assert summary["best_val_epoch"] >= 1
    assert summary["best_val_prediction_loss"] >= 0
    assert summary["best_val_total_loss"] >= 0
    assert summary["selected_checkpoint_policy"] == "best_val_total_loss"
    assert "beats_persistence_baseline" in summary
    assert "beats_action_agnostic_baseline" in summary
    assert summary["action_conditioning_utility"] in {
        "POSITIVE",
        "NEUTRAL",
        "NEGATIVE_DIAGNOSTIC_ONLY",
    }
    assert baselines["arc_lewm_action_conditioned_prediction_loss"] >= 0
    assert baselines["baseline_persistence_prediction_loss"] >= 0
    assert baselines["baseline_action_agnostic_prediction_loss"] >= 0
    assert baselines["baseline_scores_counted_as_support"] is False
    assert diagnostics["collapse_detected"] is False
    assert diagnostics["latent_variance_above_min"] is True
    assert diagnostics["nan_or_inf_detected"] is False
    assert report["split"]["mode"] == "by_game"
    assert len(report["split"]["train_games"]) == 4
    assert len(report["split"]["val_games"]) == 2
    assert set(report["per_game_validation"]) == set(report["split"]["val_games"])
    for row in report["per_game_validation"].values():
        assert row["prediction_loss"] >= 0
        assert row["support"] == 0
    assert summary["support"] == 0
    assert summary["truth_status"] == "NOT_EVALUATED_BY_M2"
    assert summary["a32_write_performed"] is False
    assert summary["a33_write_performed"] is False
    assert report["config"]["reconstruction_loss_used"] is False
    assert report["config"]["stop_gradient_used"] is False
    assert report["config"]["ema_target_encoder_used"] is False
