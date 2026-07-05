"""Offline trainer for the M2.14c ARC-LeWM foundation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .arc_lewm_dataset import by_game_split, candidate_only_metadata
from .arc_lewm_losses import collapse_diagnostics, prediction_mse, sigreg_loss
from .arc_lewm_model import (
    ARCLeWMModel,
    DEFAULT_CANVAS_SIZE,
    action_to_id,
    encode_action_args,
    pad_crop_grid,
)
from .schema import M2_TRUTH_STATUS


DEFAULT_ARC_LEWM_MODEL_OUTPUT_PATH = Path("models") / "m2_arc_lewm.pt"
DEFAULT_ARC_LEWM_TRAINING_REPORT_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "arc_lewm_training_report.json"
)
TRAINING_REPORT_SCHEMA_VERSION = "m2.arc_lewm_training_report.v1"
CANDIDATE_REVISION_STATUS = "CANDIDATE_ONLY"


class ARCLeWMTransitionDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.canvas_size = canvas_size
        self.items = [self._prepare_item(row) for row in self.rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]

    def _prepare_item(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        grid_t, mask_t = pad_crop_grid(row["grid_t"], canvas_size=self.canvas_size)
        grid_t1, mask_t1 = pad_crop_grid(row["grid_t1"], canvas_size=self.canvas_size)
        return {
            "grid_t": grid_t.to(torch.uint8),
            "mask_t": mask_t.to(torch.bool),
            "grid_t1": grid_t1.to(torch.uint8),
            "mask_t1": mask_t1.to(torch.bool),
            "action_id": torch.tensor(action_to_id(str(row.get("action", ""))), dtype=torch.long),
            "action_args": encode_action_args(
                row.get("action_args", {}),
                canvas_size=self.canvas_size,
            ),
            "hud": torch.tensor([1.0 if dict(row.get("hud", {}) or {}).get("available") else 0.0]),
            "game_id": str(row.get("game_id", "")),
        }


def collate_transition_batch(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "grid_t": torch.stack([item["grid_t"] for item in items]).long(),
        "mask_t": torch.stack([item["mask_t"] for item in items]).float(),
        "grid_t1": torch.stack([item["grid_t1"] for item in items]).long(),
        "mask_t1": torch.stack([item["mask_t1"] for item in items]).float(),
        "action_ids": torch.stack([item["action_id"] for item in items]),
        "action_args": torch.stack([item["action_args"] for item in items]),
        "hud": torch.stack([item["hud"] for item in items]),
        "game_ids": [str(item["game_id"]) for item in items],
    }


def train_arc_lewm(
    *,
    dataset_path: str | Path,
    output_model_path: str | Path = DEFAULT_ARC_LEWM_MODEL_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_ARC_LEWM_TRAINING_REPORT_OUTPUT_PATH,
    latent_dim: int = 128,
    lambda_sigreg: float = 0.1,
    use_hud: bool = False,
    split: str = "by_game",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    seed: int = 14,
    device: str | None = None,
) -> Dict[str, Any]:
    if split != "by_game":
        raise ValueError("M2.14 foundation trainer supports only split='by_game'")
    torch.manual_seed(seed)
    rows = load_transition_rows(dataset_path)
    train_rows, val_rows, split_info = split_transition_rows(rows)
    if not train_rows:
        raise ValueError("no training transitions available")
    train_dataset = ARCLeWMTransitionDataset(train_rows, canvas_size=canvas_size)
    val_dataset = ARCLeWMTransitionDataset(val_rows, canvas_size=canvas_size)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_transition_batch,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_transition_batch,
    )
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = ARCLeWMModel(latent_dim=latent_dim, use_hud=use_hud).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    per_epoch: list[Dict[str, Any]] = []
    training_steps = 0
    best_state_dict: Dict[str, torch.Tensor] | None = None
    best_val_epoch = 0
    best_val_prediction_loss = float("inf")
    best_val_total_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_metrics, steps = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            lambda_sigreg=lambda_sigreg,
            use_hud=use_hud,
            device=resolved_device,
        )
        training_steps += steps
        val_metrics = _evaluate(
            model,
            val_loader,
            lambda_sigreg=lambda_sigreg,
            use_hud=use_hud,
            device=resolved_device,
        )
        per_epoch.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        selection_metrics = val_metrics if val_rows else train_metrics
        selected_total_loss = float(selection_metrics["total_loss"])
        if selected_total_loss < best_val_total_loss:
            best_val_epoch = epoch
            best_val_prediction_loss = float(selection_metrics["prediction_loss"])
            best_val_total_loss = selected_total_loss
            best_state_dict = _state_dict_cpu(model)
    last_state_dict = _state_dict_cpu(model)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    baseline_metrics = _evaluate_with_baselines(
        model,
        val_loader if val_rows else train_loader,
        lambda_sigreg=lambda_sigreg,
        use_hud=use_hud,
        device=resolved_device,
    )
    per_game_validation = _evaluate_per_game(
        model,
        val_loader if val_rows else train_loader,
        use_hud=use_hud,
        device=resolved_device,
    )
    diagnostics = _final_diagnostics(
        model,
        val_loader if val_rows else train_loader,
        use_hud=use_hud,
        device=resolved_device,
    )
    prediction_start = float(per_epoch[0]["train"]["prediction_loss"])
    prediction_end = float(per_epoch[-1]["train"]["prediction_loss"])
    sigreg_start = float(per_epoch[0]["train"]["sigreg_loss"])
    sigreg_end = float(per_epoch[-1]["train"]["sigreg_loss"])
    diagnostics.update(
        {
            "sigreg_loss_start": sigreg_start,
            "sigreg_loss_end": sigreg_end,
            "prediction_loss_start": prediction_start,
            "prediction_loss_end": prediction_end,
        }
    )
    finite_losses = _finite_epoch_losses(per_epoch)
    prediction_loss_decreased_or_stable = bool(
        finite_losses and prediction_end <= max(prediction_start * 1.25, prediction_start + 1e-6)
    )
    baseline_diagnostics = build_baseline_diagnostics(baseline_metrics)
    report = build_training_report(
        dataset_path=dataset_path,
        output_model_path=output_model_path,
        best_model_path=_derived_checkpoint_path(output_model_path, "best_val"),
        last_model_path=_derived_checkpoint_path(output_model_path, "last"),
        latent_dim=latent_dim,
        lambda_sigreg=lambda_sigreg,
        use_hud=use_hud,
        split_info=split_info,
        canvas_size=canvas_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=str(resolved_device),
        rows=rows,
        per_epoch=per_epoch,
        diagnostics=diagnostics,
        baseline_diagnostics=baseline_diagnostics,
        per_game_validation=per_game_validation,
        training_steps=training_steps,
        prediction_loss_decreased_or_stable=prediction_loss_decreased_or_stable,
        best_val_epoch=best_val_epoch,
        best_val_prediction_loss=best_val_prediction_loss,
        best_val_total_loss=best_val_total_loss,
    )
    _write_model_state(best_state_dict or last_state_dict, output_model_path, report)
    _write_model_state(
        best_state_dict or last_state_dict,
        _derived_checkpoint_path(output_model_path, "best_val"),
        report,
    )
    _write_model_state(
        last_state_dict,
        _derived_checkpoint_path(output_model_path, "last"),
        report,
    )
    _write_report(report, report_path)
    return report


def load_transition_rows(path: str | Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, Mapping):
                rows.append(dict(data))
    return rows


def split_transition_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, list[str]]]:
    games = sorted({str(row.get("game_id", "")) for row in rows if row.get("game_id")})
    split_info = by_game_split(games)
    train_games = set(split_info["train_games"])
    val_games = set(split_info["val_games"])
    train_rows = [dict(row) for row in rows if str(row.get("game_id", "")) in train_games]
    val_rows = [dict(row) for row in rows if str(row.get("game_id", "")) in val_games]
    if not train_rows and rows:
        train_rows = [dict(row) for row in rows]
    return train_rows, val_rows, split_info


def build_training_report(
    *,
    dataset_path: str | Path,
    output_model_path: str | Path,
    best_model_path: str | Path,
    last_model_path: str | Path,
    latent_dim: int,
    lambda_sigreg: float,
    use_hud: bool,
    split_info: Mapping[str, Sequence[str]],
    canvas_size: tuple[int, int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    rows: Sequence[Mapping[str, Any]],
    per_epoch: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    baseline_diagnostics: Mapping[str, Any],
    per_game_validation: Mapping[str, Any],
    training_steps: int,
    prediction_loss_decreased_or_stable: bool,
    best_val_epoch: int,
    best_val_prediction_loss: float,
    best_val_total_loss: float,
) -> Dict[str, Any]:
    game_ids = sorted({str(row.get("game_id", "")) for row in rows if row.get("game_id")})
    guard = candidate_only_metadata()
    summary = {
        "training_completed": bool(training_steps > 0),
        "training_steps_completed": bool(training_steps > 0),
        "training_steps": int(training_steps),
        "transitions_consumed": len(rows),
        "games_total": len(game_ids),
        "split_mode": "by_game",
        "train_games": len(split_info.get("train_games", [])),
        "val_games": len(split_info.get("val_games", [])),
        "smoke_generalization_claim": False,
        "prediction_loss_decreased_or_stable": prediction_loss_decreased_or_stable,
        "sigreg_active": bool(lambda_sigreg > 0),
        **guard,
        **{key: diagnostics[key] for key in (
            "collapse_detected",
            "latent_variance_above_min",
        "nan_or_inf_detected",
        )},
        "best_val_epoch": int(best_val_epoch),
        "best_val_prediction_loss": float(best_val_prediction_loss),
        "best_val_total_loss": float(best_val_total_loss),
        "selected_checkpoint_policy": "best_val_total_loss",
        "beats_persistence_baseline": bool(
            baseline_diagnostics.get("beats_persistence_baseline", False)
        ),
        "beats_action_agnostic_baseline": bool(
            baseline_diagnostics.get("beats_action_agnostic_baseline", False)
        ),
        "action_conditioning_utility": str(
            baseline_diagnostics.get("action_conditioning_utility", "UNKNOWN")
        ),
    }
    return {
        "config": {
            "schema_version": TRAINING_REPORT_SCHEMA_VERSION,
            "dataset_path": str(dataset_path),
            "model_output_path": str(output_model_path),
            "best_val_model_output_path": str(best_model_path),
            "last_model_output_path": str(last_model_path),
            "latent_dim": latent_dim,
            "lambda_sigreg": lambda_sigreg,
            "use_hud": use_hud,
            "hud_branch_policy": "disabled_foundation" if not use_hud else "enabled_experimental",
            "split_mode": "by_game",
            "canvas_size": list(canvas_size),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
            "reward_read": False,
            "reconstruction_loss_used": False,
            "stop_gradient_used": False,
            "ema_target_encoder_used": False,
            "selected_checkpoint_policy": "best_val_total_loss",
        },
        "checkpoints": {
            "selected": str(output_model_path),
            "best_val": str(best_model_path),
            "last": str(last_model_path),
            "best_val_epoch": int(best_val_epoch),
            "best_val_prediction_loss": float(best_val_prediction_loss),
            "best_val_total_loss": float(best_val_total_loss),
            "selected_checkpoint_policy": "best_val_total_loss",
        },
        "split": {
            "mode": "by_game",
            "train_games": list(split_info.get("train_games", [])),
            "val_games": list(split_info.get("val_games", [])),
        },
        "per_epoch_losses": list(per_epoch),
        "anti_collapse_diagnostics": dict(diagnostics),
        "baseline_diagnostics": dict(baseline_diagnostics),
        "per_game_validation": dict(per_game_validation),
        "summary": summary,
        "candidate_only_metadata": guard,
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "revision_status": CANDIDATE_REVISION_STATUS,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "world_model_prediction_counted_as_evidence": False,
        "world_model_score_counted_as_support": False,
    }


def _run_epoch(
    model: ARCLeWMModel,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    lambda_sigreg: float,
    use_hud: bool,
    device: torch.device,
) -> tuple[Dict[str, float], int]:
    model.train()
    totals = {"total_loss": 0.0, "prediction_loss": 0.0, "sigreg_loss": 0.0}
    examples = 0
    steps = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        pred_loss, sig_loss = _batch_losses(
            model,
            batch,
            lambda_sigreg=lambda_sigreg,
            use_hud=use_hud,
            device=device,
        )
        total_loss = pred_loss + lambda_sigreg * sig_loss
        total_loss.backward()
        optimizer.step()
        batch_size = int(batch["grid_t"].shape[0])
        totals["total_loss"] += float(total_loss.detach().cpu().item()) * batch_size
        totals["prediction_loss"] += float(pred_loss.detach().cpu().item()) * batch_size
        totals["sigreg_loss"] += float(sig_loss.detach().cpu().item()) * batch_size
        examples += batch_size
        steps += 1
    return _mean_metrics(totals, examples), steps


def _evaluate(
    model: ARCLeWMModel,
    loader: DataLoader,
    *,
    lambda_sigreg: float,
    use_hud: bool,
    device: torch.device,
) -> Dict[str, float]:
    if len(loader.dataset) == 0:
        return {"total_loss": 0.0, "prediction_loss": 0.0, "sigreg_loss": 0.0}
    model.eval()
    totals = {"total_loss": 0.0, "prediction_loss": 0.0, "sigreg_loss": 0.0}
    examples = 0
    with torch.no_grad():
        for batch in loader:
            pred_loss, sig_loss = _batch_losses(
                model,
                batch,
                lambda_sigreg=lambda_sigreg,
                use_hud=use_hud,
                device=device,
            )
            total_loss = pred_loss + lambda_sigreg * sig_loss
            batch_size = int(batch["grid_t"].shape[0])
            totals["total_loss"] += float(total_loss.cpu().item()) * batch_size
            totals["prediction_loss"] += float(pred_loss.cpu().item()) * batch_size
            totals["sigreg_loss"] += float(sig_loss.cpu().item()) * batch_size
            examples += batch_size
    return _mean_metrics(totals, examples)


def _evaluate_with_baselines(
    model: ARCLeWMModel,
    loader: DataLoader,
    *,
    lambda_sigreg: float,
    use_hud: bool,
    device: torch.device,
) -> Dict[str, float]:
    if len(loader.dataset) == 0:
        return {
            "arc_lewm_action_conditioned_prediction_loss": 0.0,
            "baseline_persistence_prediction_loss": 0.0,
            "baseline_action_agnostic_prediction_loss": 0.0,
            "sigreg_loss": 0.0,
        }
    model.eval()
    totals = {
        "arc_lewm_action_conditioned_prediction_loss": 0.0,
        "baseline_persistence_prediction_loss": 0.0,
        "baseline_action_agnostic_prediction_loss": 0.0,
        "sigreg_loss": 0.0,
    }
    examples = 0
    with torch.no_grad():
        for batch in loader:
            components = _batch_prediction_components(
                model,
                batch,
                use_hud=use_hud,
                device=device,
            )
            z_t = components["z_t"]
            z_t1 = components["z_t1"]
            z_hat_t1 = components["z_hat_t1"]
            action_ids_agnostic = torch.zeros_like(components["action_ids"])
            args_agnostic = torch.zeros_like(components["action_args"])
            hud = components["hud"] if use_hud else None
            z_hat_agnostic = model.predictor(
                z_t,
                action_ids_agnostic,
                args_agnostic,
                hud,
            )
            batch_size = int(z_t.shape[0])
            totals["arc_lewm_action_conditioned_prediction_loss"] += (
                float(prediction_mse(z_hat_t1, z_t1).cpu().item()) * batch_size
            )
            totals["baseline_persistence_prediction_loss"] += (
                float(prediction_mse(z_t, z_t1).cpu().item()) * batch_size
            )
            totals["baseline_action_agnostic_prediction_loss"] += (
                float(prediction_mse(z_hat_agnostic, z_t1).cpu().item()) * batch_size
            )
            if lambda_sigreg > 0:
                totals["sigreg_loss"] += (
                    float(sigreg_loss(torch.cat([z_t, z_t1], dim=0)).cpu().item())
                    * batch_size
                )
            examples += batch_size
    return _mean_metrics(totals, examples)


def _evaluate_per_game(
    model: ARCLeWMModel,
    loader: DataLoader,
    *,
    use_hud: bool,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    totals: dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for batch in loader:
            components = _batch_prediction_components(
                model,
                batch,
                use_hud=use_hud,
                device=device,
            )
            per_sample = (
                components["z_hat_t1"] - components["z_t1"]
            ).pow(2).mean(dim=1)
            for game_id, loss in zip(batch["game_ids"], per_sample):
                bucket = totals.setdefault(
                    str(game_id),
                    {"prediction_loss_sum": 0.0, "transitions": 0.0},
                )
                bucket["prediction_loss_sum"] += float(loss.cpu().item())
                bucket["transitions"] += 1.0
    return {
        game_id: {
            "prediction_loss": (
                values["prediction_loss_sum"] / max(values["transitions"], 1.0)
            ),
            "transitions": int(values["transitions"]),
            "support": 0,
            "truth_status": M2_TRUTH_STATUS,
        }
        for game_id, values in sorted(totals.items())
    }


def build_baseline_diagnostics(metrics: Mapping[str, float]) -> Dict[str, Any]:
    action_conditioned = float(
        metrics.get("arc_lewm_action_conditioned_prediction_loss", 0.0)
    )
    persistence = float(metrics.get("baseline_persistence_prediction_loss", 0.0))
    action_agnostic = float(
        metrics.get("baseline_action_agnostic_prediction_loss", 0.0)
    )
    beats_persistence = action_conditioned < persistence
    beats_action_agnostic = action_conditioned < action_agnostic
    if beats_action_agnostic:
        utility = "POSITIVE"
    elif action_conditioned <= action_agnostic * 1.05:
        utility = "NEUTRAL"
    else:
        utility = "NEGATIVE_DIAGNOSTIC_ONLY"
    return {
        "arc_lewm_action_conditioned_prediction_loss": action_conditioned,
        "baseline_persistence_prediction_loss": persistence,
        "baseline_action_agnostic_prediction_loss": action_agnostic,
        "beats_persistence_baseline": bool(beats_persistence),
        "beats_action_agnostic_baseline": bool(beats_action_agnostic),
        "action_conditioning_utility": utility,
        "baseline_scores_counted_as_support": False,
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
    }


def _batch_losses(
    model: ARCLeWMModel,
    batch: Mapping[str, Any],
    *,
    lambda_sigreg: float,
    use_hud: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    components = _batch_prediction_components(
        model,
        batch,
        use_hud=use_hud,
        device=device,
    )
    z_t = components["z_t"]
    z_t1 = components["z_t1"]
    z_hat_t1 = components["z_hat_t1"]
    pred_loss = prediction_mse(z_hat_t1, z_t1)
    sig_loss = sigreg_loss(torch.cat([z_t, z_t1], dim=0)) if lambda_sigreg > 0 else z_t.sum() * 0.0
    return pred_loss, sig_loss


def _batch_prediction_components(
    model: ARCLeWMModel,
    batch: Mapping[str, Any],
    *,
    use_hud: bool,
    device: torch.device,
) -> Dict[str, torch.Tensor | None]:
    grid_t = batch["grid_t"].to(device=device)
    mask_t = batch["mask_t"].to(device=device)
    grid_t1 = batch["grid_t1"].to(device=device)
    mask_t1 = batch["mask_t1"].to(device=device)
    action_ids = batch["action_ids"].to(device=device)
    action_args = batch["action_args"].to(device=device)
    hud = batch["hud"].to(device=device) if use_hud else None
    z_t = model.encoder(grid_t, mask_t)
    z_t1 = model.encoder(grid_t1, mask_t1)
    z_hat_t1 = model.predictor(z_t, action_ids, action_args, hud)
    return {
        "z_t": z_t,
        "z_t1": z_t1,
        "z_hat_t1": z_hat_t1,
        "action_ids": action_ids,
        "action_args": action_args,
        "hud": hud,
    }


def _final_diagnostics(
    model: ARCLeWMModel,
    loader: DataLoader,
    *,
    use_hud: bool,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    latents: list[torch.Tensor] = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= 4:
                break
            grid_t = batch["grid_t"].to(device=device)
            mask_t = batch["mask_t"].to(device=device)
            grid_t1 = batch["grid_t1"].to(device=device)
            mask_t1 = batch["mask_t1"].to(device=device)
            latents.extend(
                [
                    model.encoder(grid_t, mask_t).detach().cpu(),
                    model.encoder(grid_t1, mask_t1).detach().cpu(),
                ]
            )
    if not latents:
        return collapse_diagnostics(torch.empty((0, model.latent_dim)))
    _ = use_hud
    return collapse_diagnostics(torch.cat(latents, dim=0))


def _mean_metrics(totals: Mapping[str, float], examples: int) -> Dict[str, float]:
    denom = max(examples, 1)
    return {key: float(value) / denom for key, value in totals.items()}


def _finite_epoch_losses(per_epoch: Sequence[Mapping[str, Any]]) -> bool:
    for row in per_epoch:
        for section in ("train", "val"):
            for value in dict(row.get(section, {}) or {}).values():
                if not torch.isfinite(torch.tensor(float(value))):
                    return False
    return True


def _state_dict_cpu(model: ARCLeWMModel) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _derived_checkpoint_path(output_model_path: str | Path, suffix: str) -> Path:
    path = Path(output_model_path)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _write_model_state(
    state_dict: Mapping[str, torch.Tensor],
    output_model_path: str | Path,
    report: Mapping[str, Any],
) -> None:
    path = Path(output_model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "m2.arc_lewm_model_state.v1",
            "state_dict": dict(state_dict),
            "config": report["config"],
            "candidate_only_metadata": report["candidate_only_metadata"],
            "checkpoints": report.get("checkpoints", {}),
        },
        path,
    )


def _write_report(report: Mapping[str, Any], report_path: str | Path) -> None:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train M2.14c ARC-LeWM latent transition model.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, default=DEFAULT_ARC_LEWM_MODEL_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_ARC_LEWM_TRAINING_REPORT_OUTPUT_PATH)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--lambda-sigreg", type=float, default=0.1)
    parser.add_argument("--use-hud", type=_parse_bool, default=False)
    parser.add_argument("--split", choices=("by_game",), default="by_game")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--canvas-height", type=int, default=64)
    parser.add_argument("--canvas-width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = train_arc_lewm(
        dataset_path=args.dataset,
        output_model_path=args.out_model,
        report_path=args.report,
        latent_dim=args.latent_dim,
        lambda_sigreg=args.lambda_sigreg,
        use_hud=args.use_hud,
        split=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        canvas_size=(args.canvas_height, args.canvas_width),
        seed=args.seed,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "model_path": str(args.out_model),
                "report_path": str(args.report),
                "summary": report["summary"],
                "anti_collapse_diagnostics": report["anti_collapse_diagnostics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
