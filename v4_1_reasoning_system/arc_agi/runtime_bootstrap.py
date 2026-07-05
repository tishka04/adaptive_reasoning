"""Runtime bootstrap helpers for the ARC adaptive agent."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .game_world_model import WorldModelConfig
from .reasoning_loop import LoopConfig


DEFAULT_ENABLE_ONLINE_JEPA_TRAINING = False
DEFAULT_ENABLE_ONLINE_EBM_TRAINING = False
DEFAULT_ENABLE_VISUAL_CORTEX_WARMUP = True
DEFAULT_REASONING_MODE = "full"
DEFAULT_ABLATION_STAGE = None


def resolve_checkpoint_path(ckpt_dir: Path, *candidates: str) -> Optional[Path]:
    """Return the first existing checkpoint path from a preference-ordered list."""
    for candidate in candidates:
        path = ckpt_dir / candidate
        if path.exists():
            return path
    return None


def build_adaptive_loop_config(
    *,
    device: str,
    llm_model_name: str,
    checkpoints_dir: Optional[Path] = None,
    reasoning_mode: str = DEFAULT_REASONING_MODE,
    ablation_stage: Optional[str] = DEFAULT_ABLATION_STAGE,
    enable_online_jepa_training: bool = DEFAULT_ENABLE_ONLINE_JEPA_TRAINING,
    enable_online_ebm_training: bool = DEFAULT_ENABLE_ONLINE_EBM_TRAINING,
) -> LoopConfig:
    """Build the runtime reasoning config with explicit checkpoint preferences."""
    wm_cfg = WorldModelConfig(device=device)
    ckpt_dir = checkpoints_dir or (
        Path(__file__).resolve().parents[2] / "training" / "checkpoints"
    )
    wm_ckpt = resolve_checkpoint_path(
        ckpt_dir,
        "world_model_best.pt",
        "world_model_final.pt",
        "world_model.pt",
    )
    ebm_ckpt = resolve_checkpoint_path(
        ckpt_dir,
        "ebm_best.pt",
        "ebm_final.pt",
        "ebm.pt",
    )
    return LoopConfig(
        reasoning_mode=reasoning_mode,
        ablation_stage=ablation_stage,
        explore_budget=14,
        subgoal_budget=40,
        redecompose_interval=60,
        online_update_interval=50,
        max_strategies=6,
        use_llm=True,
        llm_model_name=llm_model_name,
        llm_device=device,
        world_model_config=wm_cfg,
        world_model_checkpoint=str(wm_ckpt) if wm_ckpt is not None else None,
        ebm_checkpoint=str(ebm_ckpt) if ebm_ckpt is not None else None,
        train_jepa_online=enable_online_jepa_training,
        train_ebm_online=enable_online_ebm_training,
    )
