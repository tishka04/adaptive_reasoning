"""M2.14d-mini latent signal extraction for ARC-LeWM.

The report produced here is an interpretable priority substrate for later M2
hypothesis generation. It never counts latent scores as evidence or support.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from .arc_lewm_losses import collapse_diagnostics, prediction_mse
from .arc_lewm_model import ARCLeWMModel, DEFAULT_CANVAS_SIZE
from .arc_lewm_trainer import (
    ARCLeWMTransitionDataset,
    collate_transition_batch,
    load_transition_rows,
    split_transition_rows,
)
from .schema import M2_TRUTH_STATUS


DEFAULT_ARC_LEWM_SIGNAL_REPORT_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "arc_lewm_signal_report.json"
)
DEFAULT_ARC_LEWM_MODEL_PATH = Path("models") / "m2_arc_lewm.pt"
DEFAULT_ARC_LEWM_DATASET_PATH = Path("training") / "m2_arc_lewm_transitions.jsonl"
DEFAULT_OBJECT_WORLD_MODEL_PACKET_PATH = (
    Path("diagnostics") / "m2" / "object_world_model_invariant_packet.json"
)
SIGNAL_REPORT_SCHEMA_VERSION = "m2.arc_lewm_signal_report.v1"
CANDIDATE_REVISION_STATUS = "CANDIDATE_ONLY"
CANDIDATE_SIGNAL_FAMILIES = (
    "high_surprise_transitions",
    "low_surprise_stable_transitions",
    "action_conditioned_delta_clusters",
    "terminal_like_latent_neighborhoods",
    "proxy_completion_gap_candidates",
)


def run_arc_lewm_signal_extractor(
    *,
    model_path: str | Path = DEFAULT_ARC_LEWM_MODEL_PATH,
    dataset_path: str | Path = DEFAULT_ARC_LEWM_DATASET_PATH,
    semantic_packet_path: str | Path = DEFAULT_OBJECT_WORLD_MODEL_PACKET_PATH,
    output_path: str | Path | None = None,
    top_k: int = 16,
    batch_size: int = 128,
    max_rows: int | None = None,
    device: str | None = None,
) -> Dict[str, Any]:
    rows = load_transition_rows(dataset_path)
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        raise ValueError("no ARC-LeWM transitions available for signal extraction")
    packet = _load_json(semantic_packet_path)
    model, model_config = load_arc_lewm_model(model_path, device=device)
    _, val_rows, split_info = split_transition_rows(rows)
    scored_all, diagnostics = score_transition_rows(
        model,
        rows,
        model_config=model_config,
        batch_size=batch_size,
        device=device,
    )
    scored_val, _ = score_transition_rows(
        model,
        val_rows or rows,
        model_config=model_config,
        batch_size=batch_size,
        device=device,
    )
    report = build_signal_report(
        model_path=model_path,
        dataset_path=dataset_path,
        semantic_packet_path=semantic_packet_path,
        output_path=output_path,
        rows=rows,
        scored_all=scored_all,
        scored_val=scored_val,
        diagnostics=diagnostics,
        split_info=split_info,
        packet=packet,
        model_config=model_config,
        top_k=top_k,
    )
    if output_path is not None:
        write_arc_lewm_signal_report(report, output_path)
    return report


def load_arc_lewm_model(
    model_path: str | Path,
    *,
    device: str | None = None,
) -> tuple[ARCLeWMModel, Dict[str, Any]]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ARC-LeWM model checkpoint not found: {path}")
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(path, map_location=resolved_device, weights_only=False)
    config = dict(checkpoint.get("config", {}) if isinstance(checkpoint, Mapping) else {})
    latent_dim = int(config.get("latent_dim", 128) or 128)
    use_hud = bool(config.get("use_hud", False))
    canvas_size = tuple(config.get("canvas_size", DEFAULT_CANVAS_SIZE) or DEFAULT_CANVAS_SIZE)
    if len(canvas_size) != 2:
        canvas_size = DEFAULT_CANVAS_SIZE
    model = ARCLeWMModel(latent_dim=latent_dim, use_hud=use_hud).to(resolved_device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, {
        "latent_dim": latent_dim,
        "use_hud": use_hud,
        "canvas_size": [int(canvas_size[0]), int(canvas_size[1])],
        "device": str(resolved_device),
    }


def score_transition_rows(
    model: ARCLeWMModel,
    rows: Sequence[Mapping[str, Any]],
    *,
    model_config: Mapping[str, Any],
    batch_size: int = 128,
    device: str | None = None,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    if not rows:
        return [], collapse_diagnostics(torch.empty((0, int(model_config.get("latent_dim", 128)))))
    resolved_device = torch.device(device or str(model_config.get("device", "cpu")))
    canvas_raw = model_config.get("canvas_size", DEFAULT_CANVAS_SIZE)
    canvas_size = (int(canvas_raw[0]), int(canvas_raw[1]))
    dataset = ARCLeWMTransitionDataset(rows, canvas_size=canvas_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_transition_batch,
    )
    scored: list[Dict[str, Any]] = []
    latents: list[torch.Tensor] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            grid_t = batch["grid_t"].to(device=resolved_device)
            mask_t = batch["mask_t"].to(device=resolved_device)
            grid_t1 = batch["grid_t1"].to(device=resolved_device)
            mask_t1 = batch["mask_t1"].to(device=resolved_device)
            action_ids = batch["action_ids"].to(device=resolved_device)
            action_args = batch["action_args"].to(device=resolved_device)
            hud = batch["hud"].to(device=resolved_device) if model.use_hud else None
            z_t = model.encoder(grid_t, mask_t)
            z_t1 = model.encoder(grid_t1, mask_t1)
            z_hat_t1 = model.predictor(z_t, action_ids, action_args, hud)
            surprise = (z_hat_t1 - z_t1).pow(2).mean(dim=1)
            delta_norm = (z_t1 - z_t).norm(dim=1)
            target_norm = z_t1.norm(dim=1)
            _ = prediction_mse(z_hat_t1, z_t1)
            latents.extend([z_t.detach().cpu(), z_t1.detach().cpu()])
            for index in range(int(grid_t.shape[0])):
                row = rows[offset + index]
                scored.append(
                    {
                        **_row_ref(row),
                        "action": str(row.get("action", "")),
                        "action_args": dict(row.get("action_args", {}) or {}),
                        "available_actions_t": list(row.get("available_actions_t", []) or []),
                        "terminal_t1": bool(row.get("terminal_t1", False)),
                        "level_delta": int(row.get("level_delta", 0) or 0),
                        "latent_surprise_score": float(surprise[index].detach().cpu().item()),
                        "latent_delta_norm": float(delta_norm[index].detach().cpu().item()),
                        "latent_target_norm": float(target_norm[index].detach().cpu().item()),
                        "support": 0,
                        "truth_status": M2_TRUTH_STATUS,
                        "world_model_score_counted_as_support": False,
                    }
                )
            offset += int(grid_t.shape[0])
    diagnostics = collapse_diagnostics(torch.cat(latents, dim=0))
    return scored, diagnostics


def build_signal_report(
    *,
    model_path: str | Path,
    dataset_path: str | Path,
    semantic_packet_path: str | Path,
    output_path: str | Path | None,
    rows: Sequence[Mapping[str, Any]],
    scored_all: Sequence[Mapping[str, Any]],
    scored_val: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    split_info: Mapping[str, Sequence[str]],
    packet: Mapping[str, Any],
    model_config: Mapping[str, Any],
    top_k: int,
) -> Dict[str, Any]:
    prediction_loss_all = _mean_score(scored_all, "latent_surprise_score")
    prediction_loss_val = _mean_score(scored_val, "latent_surprise_score")
    high_surprise = _top_rows(scored_all, key="latent_surprise_score", reverse=True, limit=top_k)
    low_surprise = _top_rows(scored_all, key="latent_surprise_score", reverse=False, limit=top_k)
    action_clusters = _action_conditioned_clusters(scored_all)
    terminal_neighborhoods = _terminal_like_neighborhoods(scored_all, top_k=top_k)
    proxy_gaps = _proxy_completion_gap_candidates(scored_all, top_k=top_k)
    contract = candidate_only_contract()
    return {
        "schema_version": SIGNAL_REPORT_SCHEMA_VERSION,
        "config": {
            "model_path": str(model_path),
            "dataset_path": str(dataset_path),
            "semantic_packet_path": str(semantic_packet_path),
            "output_path": str(output_path) if output_path is not None else None,
            "top_k": int(top_k),
            "model_config": dict(model_config),
            "score_policy": "latent_prediction_error_priority_only",
            "rollout_performed": False,
            "environment_step_performed": False,
            "reward_read": False,
        },
        "latent_prediction_quality": {
            "prediction_loss_all": prediction_loss_all,
            "prediction_loss_val": prediction_loss_val,
            "collapse_detected": bool(diagnostics.get("collapse_detected", True)),
            "latent_variance_above_min": bool(
                diagnostics.get("latent_variance_above_min", False)
            ),
            "nan_or_inf_detected": bool(diagnostics.get("nan_or_inf_detected", True)),
            "latent_std_mean": float(diagnostics.get("latent_std_mean", 0.0)),
            "latent_std_min": float(diagnostics.get("latent_std_min", 0.0)),
            "latent_cov_rank_proxy": int(diagnostics.get("latent_cov_rank_proxy", 0)),
            "latent_norm_mean": float(diagnostics.get("latent_norm_mean", 0.0)),
        },
        "candidate_signal_families": list(CANDIDATE_SIGNAL_FAMILIES),
        "signals": {
            "high_surprise_transitions": high_surprise,
            "low_surprise_stable_transitions": low_surprise,
            "action_conditioned_delta_clusters": action_clusters,
            "terminal_like_latent_neighborhoods": terminal_neighborhoods,
            "proxy_completion_gap_candidates": proxy_gaps,
        },
        "semantic_context_summary": {
            "semantic_packet_loaded": True,
            "mechanistic_context_candidates": len(
                packet.get("mechanistic_context_candidates", []) or []
            ),
            "hud_counted_as_objective_signal": False,
            "relation_progress_completion_signal": False,
        },
        "split": {
            "mode": "by_game",
            "train_games": list(split_info.get("train_games", [])),
            "val_games": list(split_info.get("val_games", [])),
        },
        "summary": {
            "transitions_scored": len(scored_all),
            "source_transitions": len(rows),
            "candidate_signal_families": len(CANDIDATE_SIGNAL_FAMILIES),
            "high_surprise_transitions": len(high_surprise),
            "proxy_completion_gap_candidates": len(proxy_gaps),
            "world_model_signal_report_built": True,
            "adds_signal_beyond_g6_claim": False,
            **contract,
        },
        "contract": contract,
        **contract,
    }


def write_arc_lewm_signal_report(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_ARC_LEWM_SIGNAL_REPORT_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def candidate_only_contract() -> Dict[str, Any]:
    return {
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "revision_status": CANDIDATE_REVISION_STATUS,
        "controlled_test_required": True,
        "world_model_counted_as_evidence": False,
        "world_model_score_counted_as_support": False,
        "world_model_prediction_counted_as_evidence": False,
        "fusion_score_counted_as_support": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "ready_for_a32": False,
        "ready_for_a33": False,
    }


def _row_ref(row: Mapping[str, Any]) -> Dict[str, Any]:
    source = dict(row.get("source", {}) or {})
    return {
        "game_id": str(row.get("game_id", "")),
        "episode_id": str(row.get("episode_id", "")),
        "step": int(row.get("step", 0) or 0),
        "source_path": str(source.get("source_path", "")),
        "source_line_number": int(source.get("line_number", 0) or 0),
    }


def _mean_score(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row.get(key, 0.0)) for row in rows) / len(rows))


def _top_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    reverse: bool,
    limit: int,
) -> list[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row.get(key, 0.0)), reverse=reverse)
    return [_signal_row(row) for row in ordered[: max(limit, 0)]]


def _signal_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    step = int(row.get("step", 0) or 0)
    transition_id = (
        f"m2_14d::{row.get('game_id', 'unknown')}::"
        f"{row.get('episode_id', 'episode')}::{step:04d}"
    )
    return {
        "signal_id": transition_id,
        "source_transition_id": transition_id,
        "context_state_origin": "human_trace_frame_before",
        "replayability": "OFFLINE_TRACE_CONTEXT_ONLY",
        "game_id": str(row.get("game_id", "")),
        "episode_id": str(row.get("episode_id", "")),
        "step": step,
        "source_path": str(row.get("source_path", "")),
        "source_line_number": int(row.get("source_line_number", 0) or 0),
        "action": str(row.get("action", "")),
        "action_args": dict(row.get("action_args", {}) or {}),
        "available_actions_t": list(row.get("available_actions_t", []) or []),
        "terminal_t1": bool(row.get("terminal_t1", False)),
        "level_delta": int(row.get("level_delta", 0) or 0),
        "latent_surprise_score": float(row.get("latent_surprise_score", 0.0)),
        "latent_delta_norm": float(row.get("latent_delta_norm", 0.0)),
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "world_model_score_counted_as_support": False,
    }


def _action_conditioned_clusters(
    rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("action", ""))].append(row)
    clusters = []
    for action, action_rows in sorted(grouped.items()):
        surprises = [float(row.get("latent_surprise_score", 0.0)) for row in action_rows]
        deltas = [float(row.get("latent_delta_norm", 0.0)) for row in action_rows]
        threshold = _quantile(surprises, 0.8)
        clusters.append(
            {
                "cluster_id": f"m2_14d::action_delta::{action}",
                "action": action,
                "transitions": len(action_rows),
                "latent_surprise_mean": sum(surprises) / max(len(surprises), 1),
                "latent_delta_norm_mean": sum(deltas) / max(len(deltas), 1),
                "high_surprise_rate": (
                    sum(1 for value in surprises if value >= threshold)
                    / max(len(surprises), 1)
                ),
                "terminal_rate": (
                    sum(1 for row in action_rows if bool(row.get("terminal_t1", False)))
                    / max(len(action_rows), 1)
                ),
                "support": 0,
                "truth_status": M2_TRUTH_STATUS,
                "world_model_score_counted_as_support": False,
            }
        )
    return clusters


def _terminal_like_neighborhoods(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
) -> Dict[str, Any]:
    terminal_rows = [row for row in rows if bool(row.get("terminal_t1", False))]
    action_counts = Counter(str(row.get("action", "")) for row in terminal_rows)
    return {
        "terminal_transition_count": len(terminal_rows),
        "terminal_action_histogram": dict(sorted(action_counts.items())),
        "highest_surprise_terminal_transitions": _top_rows(
            terminal_rows,
            key="latent_surprise_score",
            reverse=True,
            limit=top_k,
        ),
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "world_model_score_counted_as_support": False,
    }


def _proxy_completion_gap_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
) -> list[Dict[str, Any]]:
    surprises = [float(row.get("latent_surprise_score", 0.0)) for row in rows]
    threshold = _quantile(surprises, 0.8)
    gap_rows = [
        row
        for row in rows
        if int(row.get("level_delta", 0) or 0) <= 0
        and not bool(row.get("terminal_t1", False))
        and float(row.get("latent_surprise_score", 0.0)) >= threshold
    ]
    return _top_rows(gap_rows, key="latent_surprise_score", reverse=True, limit=top_k)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return ordered[index]


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract candidate-only latent signals from ARC-LeWM.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_ARC_LEWM_MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_ARC_LEWM_DATASET_PATH)
    parser.add_argument(
        "--semantic-packet",
        type=Path,
        default=DEFAULT_OBJECT_WORLD_MODEL_PACKET_PATH,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_ARC_LEWM_SIGNAL_REPORT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_arc_lewm_signal_extractor(
        model_path=args.model,
        dataset_path=args.dataset,
        semantic_packet_path=args.semantic_packet,
        output_path=args.out,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "latent_prediction_quality": payload["latent_prediction_quality"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
