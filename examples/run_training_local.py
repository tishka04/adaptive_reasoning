"""
Local training pipeline — runs Phases 2-4 on CPU without LLM.

Demonstrates:
  Phase 2: Rule-based router → collect rollouts on synthetic problems
  Phase 3: Train JEPA-lite world model on collected trajectories
  Phase 4: Train EBM router on (good, bad) reasoning action pairs

Usage:
    python examples/run_training_local.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader, random_split

from v4_1_reasoning_system.orchestration.controller import ReasoningController
from v4_1_reasoning_system.training.collect_rollouts import RolloutCollector
from v4_1_reasoning_system.training.train_world_model import (
    WorldModelTrainer,
    WorldModelDataset,
    build_world_model_dataset,
)
from v4_1_reasoning_system.training.train_router import train_router_from_buffer


LATENT_DIM = 64
ACTION_DIM = 16
DEVICE = "cpu"


def main():
    t0 = time.time()

    # ==================================================================
    # Phase 2: Rule-based router — Collect rollouts
    # ==================================================================
    print("=" * 60)
    print("  PHASE 2: Collect rollouts with rule-based router")
    print("=" * 60)

    controller = ReasoningController(
        use_learned_router=False,
        max_iterations=5,
        max_time_seconds=30.0,
        device=DEVICE,
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
    )

    problems = RolloutCollector.generate_benchmark_suite(
        n_planning=15,
        n_scheduling=15,
        n_optimization=15,
        n_coding=0,
    )
    print(f"Generated {len(problems)} benchmark problems")

    collector = RolloutCollector(controller)
    results = collector.collect(problems, verbose=True)

    successes = sum(1 for r in results if r["valid"])
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\nPhase 2 results: {successes}/{len(results)} valid, avg score={avg_score:.3f}")

    stats = controller.replay_buffer.statistics()
    print(f"Replay buffer: {stats['count']} trajectories")
    print(f"Mode usage: {stats.get('mode_counts', {})}")

    # ==================================================================
    # Phase 3: Train JEPA-lite world model
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PHASE 3: Train world model (transition predictor + aux heads)")
    print("=" * 60)

    wm_records = build_world_model_dataset(
        controller.replay_buffer,
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
    )
    print(f"World model dataset: {len(wm_records)} records")

    if len(wm_records) >= 10:
        dataset = WorldModelDataset(wm_records)
        val_size = max(1, int(len(dataset) * 0.2))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        wm_trainer = WorldModelTrainer(
            predictor=controller.transition_predictor,
            aux_heads=controller.aux_heads,
            lr=3e-4,
            device=DEVICE,
        )

        NUM_WM_EPOCHS = 50
        for epoch in range(NUM_WM_EPOCHS):
            train_m = wm_trainer.train_epoch(train_loader)
            if (epoch + 1) % 10 == 0:
                val_m = wm_trainer.evaluate(val_loader)
                print(
                    f"  Epoch {epoch+1:3d}/{NUM_WM_EPOCHS}: "
                    f"loss={train_m['loss']:.4f} "
                    f"latent={train_m['latent_loss']:.4f} "
                    f"aux={train_m['aux_loss']:.4f} | "
                    f"val_latent={val_m['val_latent_loss']:.4f} "
                    f"val_acc={val_m['val_validity_acc']:.3f}"
                )

        print("World model training complete.")
    else:
        print(f"Skipping world model training (only {len(wm_records)} records, need >= 10)")

    # ==================================================================
    # Phase 4: Train EBM router
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PHASE 4: Train EBM router (ranking loss)")
    print("=" * 60)

    router_result = train_router_from_buffer(
        router=controller.ebm_router,
        replay_buffer=controller.replay_buffer,
        num_epochs=50,
        batch_size=16,
        lr=1e-4,
        margin=1.0,
        device=DEVICE,
        verbose=True,
    )

    print(f"\nRouter training: {router_result['status']}")
    if router_result["status"] == "trained":
        print(f"  Pairs: {router_result['num_pairs']}")
        print(f"  Best val accuracy: {router_result['best_val_accuracy']:.3f}")

    # ==================================================================
    # Phase 5: Evaluate with learned router
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PHASE 5: Evaluate with learned EBM router")
    print("=" * 60)

    if router_result["status"] == "trained":
        controller.use_learned_router = True
        print("Switched to learned EBM router")
    else:
        print("Insufficient training data, staying on rule-based router")

    test_problems = RolloutCollector.generate_benchmark_suite(
        n_planning=5,
        n_scheduling=5,
        n_optimization=5,
        n_coding=0,
    )

    test_results = collector.collect(test_problems, verbose=True)
    test_successes = sum(1 for r in test_results if r["valid"])
    test_avg = sum(r["score"] for r in test_results) / len(test_results)

    print(f"\nLearned router: {test_successes}/{len(test_results)} valid, avg={test_avg:.3f}")

    # ==================================================================
    # Comparison
    # ==================================================================
    print("\n" + "=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    print(f"  Rule-based  : {successes}/{len(results)} valid, avg score={avg_score:.3f}")
    print(f"  Learned EBM : {test_successes}/{len(test_results)} valid, avg score={test_avg:.3f}")

    # ==================================================================
    # Save checkpoint
    # ==================================================================
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "local_trained")
    controller.save_checkpoint(ckpt_dir)
    print(f"\nCheckpoint saved to: {os.path.abspath(ckpt_dir)}")

    total_time = time.time() - t0
    print(f"\nTotal training time: {total_time:.1f}s")

    final_stats = controller.replay_buffer.statistics()
    print(f"Final replay buffer: {final_stats['count']} trajectories")
    print(f"Overall success rate: {final_stats.get('success_rate', 0):.0%}")


if __name__ == "__main__":
    main()
