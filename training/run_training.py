"""
End-to-end training pipeline for JEPA World Model + EBM Scorer.

Steps:
  1. Collect transitions from all games (smart-random agent)
  2. Train JEPA world model on transitions
  3. Train EBM scorer using frozen world model encoder

Usage:
    python training/run_training.py [--skip-collect] [--device cpu]

All outputs go to training/data/ and training/checkpoints/.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run_step(name: str, cmd: list[str]) -> bool:
    """Run a subprocess and return True if it succeeds."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {name}")
    logger.info(f"{'='*60}")
    logger.info(f"Command: {' '.join(cmd)}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"FAILED: {name} (exit code {result.returncode}, {elapsed:.1f}s)")
        return False

    logger.info(f"DONE: {name} ({elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Full training pipeline")
    parser.add_argument("--skip-collect", action="store_true", help="Skip data collection")
    parser.add_argument("--actions", type=int, default=300, help="Actions per game per run")
    parser.add_argument("--runs", type=int, default=3, help="Runs per game")
    parser.add_argument("--wm-epochs", type=int, default=50, help="World model training epochs")
    parser.add_argument("--ebm-epochs", type=int, default=30, help="EBM training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    python = sys.executable
    data_dir = str(PROJECT_ROOT / "training" / "data")
    ckpt_dir = str(PROJECT_ROOT / "training" / "checkpoints")

    total_t0 = time.time()

    # Step 1: Collect transitions
    if not args.skip_collect:
        ok = run_step("Collect Transitions", [
            python, str(PROJECT_ROOT / "training" / "collect_transitions.py"),
            "--actions", str(args.actions),
            "--runs", str(args.runs),
            "--out", data_dir,
        ])
        if not ok:
            sys.exit(1)
    else:
        logger.info("Skipping data collection (--skip-collect)")

    # Step 2: Train world model
    ok = run_step("Train JEPA World Model", [
        python, str(PROJECT_ROOT / "training" / "train_world_model.py"),
        "--data", data_dir,
        "--epochs", str(args.wm_epochs),
        "--batch", str(args.batch),
        "--device", args.device,
        "--out", ckpt_dir,
    ])
    if not ok:
        sys.exit(1)

    # Step 3: Train EBM
    ok = run_step("Train EBM Scorer", [
        python, str(PROJECT_ROOT / "training" / "train_ebm.py"),
        "--data", data_dir,
        "--wm", str(Path(ckpt_dir) / "world_model_best.pt"),
        "--epochs", str(args.ebm_epochs),
        "--batch", str(args.batch),
        "--device", args.device,
        "--out", ckpt_dir,
    ])
    if not ok:
        sys.exit(1)

    total_elapsed = time.time() - total_t0
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL DONE in {total_elapsed:.1f}s")
    logger.info(f"Checkpoints: {ckpt_dir}")
    logger.info(f"  - world_model_best.pt")
    logger.info(f"  - ebm_best.pt")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
