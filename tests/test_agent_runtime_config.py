from pathlib import Path

from v4_1_reasoning_system.arc_agi.runtime_bootstrap import (
    DEFAULT_ABLATION_STAGE,
    DEFAULT_ENABLE_ONLINE_EBM_TRAINING,
    DEFAULT_ENABLE_ONLINE_JEPA_TRAINING,
    DEFAULT_ENABLE_VISUAL_CORTEX_WARMUP,
    DEFAULT_REASONING_MODE,
    build_adaptive_loop_config,
    resolve_checkpoint_path,
)


def test_checkpoint_resolver_prefers_best_then_final(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    (ckpt_dir / "world_model_final.pt").write_text("final")
    resolved = resolve_checkpoint_path(
        ckpt_dir,
        "world_model_best.pt",
        "world_model_final.pt",
        "world_model.pt",
    )
    assert resolved == ckpt_dir / "world_model_final.pt"

    (ckpt_dir / "world_model_best.pt").write_text("best")
    resolved = resolve_checkpoint_path(
        ckpt_dir,
        "world_model_best.pt",
        "world_model_final.pt",
        "world_model.pt",
    )
    assert resolved == ckpt_dir / "world_model_best.pt"


def test_loop_config_uses_pretrained_checkpoints_and_explicit_training_flags():
    checkpoints_dir = Path(__file__).resolve().parents[1] / "training" / "checkpoints"
    cfg = build_adaptive_loop_config(
        device="cpu",
        llm_model_name="Qwen/Qwen2.5-3B-Instruct",
        checkpoints_dir=checkpoints_dir,
    )

    assert cfg.world_model_checkpoint is not None
    assert cfg.world_model_checkpoint.endswith("world_model_best.pt")
    assert cfg.ebm_checkpoint is not None
    assert cfg.ebm_checkpoint.endswith("ebm_best.pt")

    assert cfg.train_jepa_online is DEFAULT_ENABLE_ONLINE_JEPA_TRAINING
    assert cfg.train_ebm_online is DEFAULT_ENABLE_ONLINE_EBM_TRAINING
    assert DEFAULT_ENABLE_VISUAL_CORTEX_WARMUP is True
    assert DEFAULT_REASONING_MODE == "full"
    assert DEFAULT_ABLATION_STAGE is None
    assert cfg.reasoning_mode == "full"
    assert cfg.ablation_stage == "jepa_ebm"
    assert cfg.enabled_features()["jepa_ebm_rerank"] is True
    assert cfg.world_model_config.device == "cpu"
    assert cfg.sampler_stage == "v0"
    assert cfg.planner_mode == "prior"
    assert cfg.enable_trajectory_continuation is False


def test_symbolic_core_defaults_to_short_horizon_without_learned_rerank():
    cfg = build_adaptive_loop_config(
        device="cpu",
        llm_model_name="Qwen/Qwen2.5-3B-Instruct",
        checkpoints_dir=Path(__file__).resolve().parents[1] / "training" / "checkpoints",
        reasoning_mode="symbolic_core",
    )

    assert cfg.reasoning_mode == "symbolic_core"
    assert cfg.ablation_stage == "short_horizon"
    assert cfg.train_jepa_online is False
    assert cfg.train_ebm_online is False
    assert cfg.enabled_features() == {
        "symbolic_observer": True,
        "game_memory_policy": True,
        "goal_pursuit": True,
        "trajectory_memory": True,
        "short_horizon_sampling": True,
        "visual_cortex": False,
        "jepa_ebm_rerank": False,
    }
