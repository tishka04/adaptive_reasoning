"""SAGE.11 pilot, compact model, loss, gate, and adaptation tests."""

from __future__ import annotations

import numpy as np
import torch

from theory.sage11.adaptation import (
    OnlineAdaptationConfig,
    OnlineDynamicsAdapter,
)
from theory.sage11.bridge import WorldModelActionPredictor
from theory.sage11.authority import NeuralActionCandidate
from theory.live_transition_loop import build_observation
from theory.sage11.model import Sage11GraphWorldModel, WorldModelConfig
from theory.sage11.pilot import run_effect_predictability_pilot
from theory.sage11.evaluation import (
    CheckpointedRunLog,
    PairedRunResult,
    ShadowGateMetrics,
    holdout_promotion_report,
    shadow_gate_report,
)
from theory.sage11.splits import NEURO_HOLDOUT_V1
from theory.sage11.training import (
    Sage11WorldModelTrainer,
    TrainerConfig,
    WorldModelGateMetrics,
    evaluate_world_model_gates,
    world_model_loss,
)


def test_graph_world_model_stays_below_parameter_cap_and_gates_terminal_head():
    model = Sage11GraphWorldModel()
    assert model.parameter_count < 5_000_000
    atom_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    atom_mask = atom_ids != 0
    action = torch.zeros((2, 6))
    disabled = model(atom_ids, atom_mask, action)
    assert torch.all(disabled["terminal_logit"] < -10)
    assert disabled["effect_logits"].shape == (2, 64)
    model.set_strong_terminal_events(100)
    enabled = model(atom_ids, atom_mask, action)
    assert model.terminal_head_enabled
    assert enabled["next_latent"].shape == (2, 128)
    assert model.checkpoint_metadata()["legacy_weights_loaded"] is False


def test_world_model_loss_downweights_weak_progress_and_is_finite():
    model = Sage11GraphWorldModel(WorldModelConfig(effect_classes=4))
    model.set_strong_terminal_events(100)
    atom_ids = torch.tensor([[1, 2], [3, 4]])
    output = model(atom_ids, atom_ids != 0, torch.zeros((2, 6)))
    targets = {
        "next_latent": torch.randn((2, 128)),
        "shuffled_next_latent": torch.randn((2, 128)),
        "effect_class": torch.tensor([0, 1]),
        "changed": torch.tensor([1.0, 1.0]),
        "progress": torch.tensor([1.0, 1.0]),
        "progress_is_weak": torch.tensor([False, True]),
        "terminal": torch.tensor([1.0, 0.0]),
        "risk": torch.tensor([0.0, 0.0]),
        "noop": torch.tensor([0.0, 0.0]),
    }
    losses = world_model_loss(
        output,
        targets,
        terminal_head_enabled=True,
    )
    assert torch.isfinite(losses["total"])
    assert losses["terminal"].item() > 0.0


def test_concrete_trainer_updates_bootstrap_heads_and_emits_checkpoint():
    model = Sage11GraphWorldModel(WorldModelConfig(
        vocab_size=128,
        effect_classes=4,
        hidden_size=32,
        latent_size=16,
        message_layers=1,
        bootstrap_heads=2,
    ))
    trainer = Sage11WorldModelTrainer(
        model,
        config=TrainerConfig(device="cpu"),
    )
    batch = {
        "atom_ids": torch.tensor([
            [1, 2, 0],
            [3, 4, 0],
            [5, 6, 7],
            [8, 9, 0],
        ]),
        "atom_mask": torch.tensor([
            [True, True, False],
            [True, True, False],
            [True, True, True],
            [True, True, False],
        ]),
        "next_atom_ids": torch.tensor([
            [2, 3, 0],
            [4, 5, 0],
            [6, 7, 8],
            [9, 10, 0],
        ]),
        "next_atom_mask": torch.tensor([
            [True, True, False],
            [True, True, False],
            [True, True, True],
            [True, True, False],
        ]),
        "action": torch.zeros((4, 6)),
        "effect_class": torch.tensor([0, 1, 2, 3]),
        "changed": torch.tensor([1.0, 1.0, 0.0, 1.0]),
        "progress": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "progress_is_weak": torch.tensor([False, True, False, False]),
        "terminal": torch.zeros(4),
        "risk": torch.zeros(4),
        "noop": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }
    outcome = trainer.train_step(batch)
    assert outcome["bootstrap"] > 0.0
    assert trainer.checkpoint()["metadata"]["steps"] == 1


def test_effect_pilot_detects_clear_predictable_structure():
    generator = np.random.default_rng(11)
    feature = generator.normal(size=(200, 4))
    labels = (feature[:, 0] > 0).astype(np.int64)
    mask = np.arange(len(labels)) % 5 != 0
    result = run_effect_predictability_pilot(
        feature,
        labels,
        train_mask=mask,
    )
    assert result.go
    assert result.classifier_macro_f1 > result.majority_macro_f1


def test_amended_world_model_gates_pass_only_source_validation():
    metrics = WorldModelGateMetrics(
        changed_transition_accuracy=0.80,
        persistence_changed_accuracy=0.60,
        action_shuffle_degradation=0.12,
        effect_macro_f1=0.70,
        effect_majority_macro_f1=0.50,
        risk_ece=0.08,
        noop_ece=0.09,
        latent_feature_std=0.2,
        validation_games=("re86", "ls20", "sc25"),
    )
    assert evaluate_world_model_gates(metrics).passed
    contaminated = WorldModelGateMetrics(
        **{
            **metrics.__dict__,
            "validation_games": ("re86", "s5i5"),
        }
    )
    assert not evaluate_world_model_gates(contaminated).passed


def test_online_adapter_updates_every_32_steps_with_at_most_four_gradients():
    adapter = OnlineDynamicsAdapter(
        latent_size=8,
        action_size=3,
        config=OnlineAdaptationConfig(
            update_interval=32,
            maximum_gradient_steps=4,
        ),
    )
    for _ in range(31):
        outcome = adapter.observe(
            torch.randn(8),
            torch.randn(3),
            torch.randn(8),
        )
        assert outcome["last_update_steps"] == 0
    outcome = adapter.observe(
        torch.randn(8),
        torch.randn(3),
        torch.randn(8),
    )
    assert outcome["last_update_steps"] == 4
    assert adapter.summary()["replay_size"] == 32
    adapter.reset_game_seed()
    assert adapter.summary()["replay_size"] == 0
    assert adapter.summary()["gradient_steps"] == 0


def test_world_model_bridge_emits_zero_support_typed_hypotheses():
    model = Sage11GraphWorldModel()
    observation = build_observation(
        np.zeros((4, 4), dtype=np.int32),
        available_actions=("ACTION1", "ACTION2"),
    )
    predictor = WorldModelActionPredictor(model, device="cpu")
    predictions = predictor(
        observation,
        (
            NeuralActionCandidate("ACTION1"),
            NeuralActionCandidate("ACTION2"),
        ),
    )
    assert len(predictions) == 2
    assert all(
        atom.support == 0
        for prediction in predictions
        for atom in prediction.hypotheses
    )


def test_shadow_and_complete_holdout_promotion_gates(tmp_path):
    shadow = shadow_gate_report(ShadowGateMetrics(
        action_identity_mismatches=0,
        would_be_successful_route_preemptions=0,
        neural_top1_productivity=0.7,
        symbolic_top1_productivity=0.5,
        neural_top3_productivity=0.9,
        symbolic_top3_productivity=0.8,
        risk_ece=0.05,
        noop_ece=0.06,
        inference_peak_ms=8.0,
        inference_budget_ms=10.0,
    ))
    assert shadow["passed"]
    runs = [
        PairedRunResult(
            game_id=game,
            seed=seed,
            active_score=2.0,
            off_score=1.0,
            active_levels=1,
            off_levels=0,
            active_wins=1,
            off_wins=0,
            active_digest=f"active-{game}-{seed}",
            off_digest=f"off-{game}-{seed}",
        )
        for game in NEURO_HOLDOUT_V1
        for seed in range(5)
    ]
    assert holdout_promotion_report(runs)["passed"]
    log = CheckpointedRunLog(tmp_path / "runs.json")
    key = log.run_key(
        game_id="s5i5",
        seed=0,
        arm="off",
        budget=4000,
        resets=14,
    )
    log.record(key, {"score": 1.0})
    assert CheckpointedRunLog(tmp_path / "runs.json").completed(key)
