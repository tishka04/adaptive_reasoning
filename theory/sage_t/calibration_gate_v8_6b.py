"""SAGE.T8.6b channel-selective calibration on frozen T8.6 schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from theory.sage12.bound_mechanic_pilot import load_pairs

from . import calibration_gate_v8_6 as v86
from .executor import ProgramExecutor
from .posterior_v3 import (
    T8_6B_POLICIES,
    ChannelCalibratedProgramPosterior,
    ChannelPosteriorUpdatePolicy,
)

FORMAT_VERSION = "sage-t8.6b-channel-calibration-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t8_6b_selection_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "calibration_v8_6b"
DEFAULT_PARENT_MANIFEST = v86.DEFAULT_SELECTION_MANIFEST
DEFAULT_PARENT_REPORT = v86.DEFAULT_OUTPUT_DIR / "selection_report.json"
DEFAULT_ACTION_SCHEDULES = v86.DEFAULT_OUTPUT_DIR / "action_schedules.json"
CHALLENGERS = (
    "terminal_tempered",
    "teleology_tempered",
    "teleology_correlation_aware",
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _checksum(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(_json_safe(value)).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical(_json_safe(row)) + "\n")
    os.replace(temporary, path)


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: _file_sha256(directory / name)
        for name in ("posterior_v3.py", "calibration_gate_v8_6b.py")
    }


def _load_parent_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86._checksum(unsigned):
        raise ValueError("parent T8.6 report checksum mismatch")
    if payload.get("status") != "T8_6_SELECTION_FAILED_CLOSED":
        raise ValueError("T8.6b requires the frozen failed T8.6 selection")
    if payload.get("selected_challenger") is not None:
        raise ValueError("T8.6 already selected a challenger")
    if payload.get("source_validation_authorized") is not False:
        raise ValueError("T8.6 source-validation firewall is open")
    return payload


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    parent_manifest_path: str | Path = DEFAULT_PARENT_MANIFEST,
    parent_report_path: str | Path = DEFAULT_PARENT_REPORT,
    schedules_path: str | Path = DEFAULT_ACTION_SCHEDULES,
) -> dict[str, Any]:
    parent_manifest = v86.load_selection_manifest(parent_manifest_path)
    parent_report = _load_parent_report(parent_report_path)
    schedule_file = Path(schedules_path)
    schedules = json.loads(schedule_file.read_text(encoding="utf-8"))
    if len(schedules) != int(parent_manifest["corpus"]["roots"]):
        raise ValueError("T8.6 action schedules do not cover every root")
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T8_6B_SOURCE_TRAIN_SELECTION",
        "frozen_at": "2026-08-06",
        "parent_t8_6_manifest_checksum": parent_manifest["manifest_checksum"],
        "parent_t8_6_scientific_checksum": parent_report[
            "scientific_checksum"
        ],
        "action_schedules_sha256": _file_sha256(schedule_file),
        "code_sha256": _code_hashes(),
        "source_train_games": list(v86.EXPECTED_GAMES),
        "forbidden_games": sorted(v86.FORBIDDEN_GAMES),
        "corpus": dict(v86.EXPECTED_CORPUS),
        "checkpoints": list(v86.CHECKPOINTS),
        "policies": {
            name: _json_safe(asdict(policy))
            for name, policy in T8_6B_POLICIES.items()
        },
        "bootstrap": {"samples": 10_000, "seed": 8606, "confidence": 0.95},
        "selection": dict(parent_manifest["selection"]),
        "confirmation": dict(parent_manifest["confirmation"]),
        "frozen_invariants": {
            "dsl": "unchanged",
            "executor": "unchanged",
            "mdl_priors": "unchanged",
            "channel_weights": "unchanged",
            "decision_utility": "unchanged",
            "actions": "exact_t8_6_schedules",
            "only_inference_change": "channel_specific_evidence_multiplier",
        },
        "firewall": dict(parent_manifest["firewall"]),
    }
    payload["manifest_checksum"] = _checksum(payload)
    _write_json(Path(output_path), payload)
    return payload


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    parent_manifest_path: str | Path = DEFAULT_PARENT_MANIFEST,
    parent_report_path: str | Path = DEFAULT_PARENT_REPORT,
    schedules_path: str | Path = DEFAULT_ACTION_SCHEDULES,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != _checksum(unsigned):
        raise ValueError("SAGE.T8.6b manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported SAGE.T8.6b manifest")
    if payload.get("status") != "FROZEN_BEFORE_T8_6B_SOURCE_TRAIN_SELECTION":
        raise ValueError("SAGE.T8.6b manifest is not frozen")
    parent_manifest = v86.load_selection_manifest(parent_manifest_path)
    parent_report = _load_parent_report(parent_report_path)
    if payload.get("parent_t8_6_manifest_checksum") != parent_manifest.get(
        "manifest_checksum"
    ):
        raise ValueError("T8.6b parent manifest drifted")
    if payload.get("parent_t8_6_scientific_checksum") != parent_report.get(
        "scientific_checksum"
    ):
        raise ValueError("T8.6b parent scientific result drifted")
    if payload.get("corpus") != v86.EXPECTED_CORPUS:
        raise ValueError("T8.6b corpus contract drifted")
    if set(payload.get("source_train_games", ())) & v86.FORBIDDEN_GAMES:
        raise ValueError("T8.6b firewall violation")
    firewall = payload.get("firewall", {})
    if firewall.get("authority") != "shadow" or any(
        firewall.get(key) is not False
        for key in ("source_validation_opened", "holdout_opened", "ar25_opened")
    ):
        raise ValueError("T8.6b firewall is open")
    if verify_hashes:
        if payload.get("code_sha256") != _code_hashes():
            raise ValueError("T8.6b code drifted")
        if payload.get("action_schedules_sha256") != _file_sha256(
            Path(schedules_path)
        ):
            raise ValueError("T8.6b action schedules drifted")
    return payload


def _new_channel_posterior(
    policy: ChannelPosteriorUpdatePolicy,
    *,
    executor: ProgramExecutor,
    manifest: Mapping[str, Any],
    dynamics_only: bool = False,
) -> ChannelCalibratedProgramPosterior:
    config = manifest["posterior"]
    return ChannelCalibratedProgramPosterior(
        executor=executor,
        update_policy=policy,
        maximum_particles=int(config["maximum_particles"]),
        channel_weights=v86._weights(
            "dynamics_only" if dynamics_only else "joint"
        ),
        unknown_coverage_penalty=float(config["unknown_coverage_penalty"]),
        repair_ess_threshold=float(config["repair_ess_threshold"]),
        repair_log_likelihood_threshold=float(
            config["repair_log_likelihood_threshold"]
        ),
    )


@contextmanager
def _channel_runner() -> Iterable[None]:
    original = v86._new_posterior
    v86._new_posterior = _new_channel_posterior
    try:
        yield
    finally:
        v86._new_posterior = original


def _collapse_rate(
    updates: Sequence[Mapping[str, Any]],
    condition: str,
) -> float:
    selected = [row for row in updates if row["condition"] == condition]
    if not selected:
        return float("nan")
    return mean(float(bool(row.get("semantic_collapse"))) for row in selected)


def select_challenger(
    rows: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    config = manifest["selection"]
    bootstrap = manifest["bootstrap"]
    legacy_collapse = _collapse_rate(updates, "legacy")
    evaluations = {}
    survivors = []
    for name in CHALLENGERS:
        collapse = _collapse_rate(updates, name)
        collapse_ok = collapse <= float(
            config["maximum_semantic_collapse_rate"]
        ) and (
            legacy_collapse <= 0.0
            or collapse
            <= legacy_collapse
            * (1.0 - float(config["minimum_collapse_reduction_fraction"]))
        )
        metrics: dict[str, Any] = {
            "collapse_rate": collapse,
            "legacy_collapse_rate": legacy_collapse,
        }
        interval_checks = []
        for metric in ("terminal_brier", "terminal_log_loss"):
            legacy = v86._root_metric(rows, condition="legacy", metric=metric)
            challenger = v86._root_metric(rows, condition=name, metric=metric)
            interval = v86.paired_bootstrap_interval(
                legacy,
                challenger,
                samples=int(bootstrap["samples"]),
                seed=int(bootstrap["seed"]),
            )
            metrics[f"{metric}_improvement"] = asdict(interval)
            interval_checks.append(
                interval.lower
                > float(config["paired_interval_lower_must_exceed"])
            )
        game_likelihood = {}
        for game in v86.EXPECTED_GAMES:
            legacy = v86._root_metric(
                rows,
                condition="legacy",
                metric="hidden_log_likelihood",
                game=game,
            )
            challenger = v86._root_metric(
                rows,
                condition=name,
                metric="hidden_log_likelihood",
                game=game,
            )
            shared = sorted(set(legacy) & set(challenger))
            game_likelihood[game] = v86._finite_mean(
                [challenger[key] - legacy[key] for key in shared]
            )
        likelihood_ok = all(
            value >= float(config["hidden_log_likelihood_game_delta_minimum"])
            for value in game_likelihood.values()
        )
        metrics["hidden_log_likelihood_delta_by_game"] = game_likelihood
        metrics["checks"] = {
            "collapse": collapse_ok,
            "terminal_brier": interval_checks[0],
            "terminal_log_loss": interval_checks[1],
            "hidden_log_likelihood_both_games": likelihood_ok,
        }
        metrics["passed"] = all(metrics["checks"].values())
        evaluations[name] = metrics
        if metrics["passed"]:
            terminal_loss = v86._finite_mean(
                list(
                    v86._root_metric(
                        rows,
                        condition=name,
                        metric="terminal_log_loss",
                    ).values()
                )
            )
            hidden_ll = v86._finite_mean(
                list(
                    v86._root_metric(
                        rows,
                        condition=name,
                        metric="hidden_log_likelihood",
                    ).values()
                )
            )
            latency = v86._finite_mean(
                [
                    float(row["decision_latency_ms"])
                    for row in rows
                    if row["condition"] == name and row["checkpoint"] == 5
                ]
            )
            survivors.append((terminal_loss, -hidden_ll, latency, name))
    return (None if not survivors else min(survivors)[-1], evaluations)


def _diagnostic_conclusion(
    rows: Sequence[Mapping[str, Any]],
    winner: str | None,
    offline_confirmation: Mapping[str, Any] | None,
) -> str:
    selected = [
        str(row.get("diagnosis"))
        for row in rows
        if row.get("condition") == (winner or "legacy")
    ]
    if selected.count("GENERATOR_MISS") > len(selected) / 2:
        return "GENERATOR_LIMITED"
    if selected.count("POSTERIOR_MISS") > len(selected) / 2:
        return "SELECTION_LIMITED"
    if winner is None:
        return "INCONCLUSIVE_FAIL_CLOSED"
    if not bool((offline_confirmation or {}).get("passed")):
        return "REPAIR_LIMITED"
    return "CALIBRATION_RECOVERED"


def run_selection(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    shard_dir: str | Path = v86.DEFAULT_SHARD_DIR,
    schedules_path: str | Path = DEFAULT_ACTION_SCHEDULES,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    maximum_roots: int | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, schedules_path=schedules_path)
    t7 = v86.load_t7_manifest(verify_code=True)
    pairs = load_pairs(str(shard_dir), v86.EXPECTED_GAMES)
    inventory = v86.corpus_inventory(pairs)
    if inventory != v86.EXPECTED_CORPUS:
        raise ValueError(f"T8.6b corpus integrity failure: {inventory}")
    episodes = v86._all_root_episodes(pairs)
    if maximum_roots is not None:
        episodes = episodes[: max(0, int(maximum_roots))]
    schedules = json.loads(Path(schedules_path).read_text(encoding="utf-8"))
    destination = Path(output_dir)
    checkpoint_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    full_arm_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with _channel_runner():
        for episode in episodes:
            if episode.episode_id not in schedules:
                raise ValueError("T8.6b encountered an unscheduled root")
            keys = [str(value) for value in schedules[episode.episode_id]]
            for policy in T8_6B_POLICIES.values():
                rows, updates, observed_keys, arms = v86._run_or_resume_episode(
                    episode,
                    policy=policy,
                    t7_manifest=t7,
                    selection_manifest=manifest,
                    output_dir=destination,
                    forced_keys=keys,
                )
                if observed_keys != keys:
                    raise RuntimeError("T8.6b action sequence drifted")
                checkpoint_rows.extend(rows)
                update_rows.extend(updates)
                full_arm_rows.extend(arms)
        teacher_rows = v86._run_teacher_shocks(
            pairs,
            manifest=t7,
            policies=T8_6B_POLICIES,
        )
    winner, challenger_evaluations = select_challenger(
        checkpoint_rows,
        update_rows,
        manifest=manifest,
    )
    confirmation_checkpoint_rows: list[dict[str, Any]] = []
    confirmation_update_rows: list[dict[str, Any]] = []
    confirmation_full_arm_rows: list[dict[str, Any]] = []
    confirmation_teacher_rows: list[dict[str, Any]] = []
    offline_confirmation: dict[str, Any] | None = None
    if winner is not None:
        repair_policy = T8_6B_POLICIES[winner].with_repair_v2()
        with _channel_runner():
            for episode in episodes:
                keys = [str(value) for value in schedules[episode.episode_id]]
                for dynamics_only in (False, True):
                    rows, updates, observed_keys, arms = (
                        v86._run_or_resume_episode(
                            episode,
                            policy=repair_policy,
                            t7_manifest=t7,
                            selection_manifest=manifest,
                            output_dir=destination,
                            forced_keys=keys,
                            dynamics_only=dynamics_only,
                        )
                    )
                    if observed_keys != keys:
                        raise RuntimeError("T8.6b Repair V2 schedule drifted")
                    confirmation_checkpoint_rows.extend(rows)
                    confirmation_update_rows.extend(updates)
                    confirmation_full_arm_rows.extend(arms)
            confirmation_teacher_rows = v86._run_teacher_shocks(
                pairs,
                manifest=t7,
                policies={repair_policy.name: repair_policy},
            )
        offline_confirmation = v86._offline_confirmation_checks(
            confirmation_checkpoint_rows,
            confirmation_update_rows,
            confirmation_full_arm_rows,
            confirmation_teacher_rows,
            joint_condition=repair_policy.name,
            dynamics_condition=f"{repair_policy.name}_dynamics_only",
            manifest=manifest,
        )
    failure_taxonomy = {
        diagnosis: sum(row["diagnosis"] == diagnosis for row in checkpoint_rows)
        for diagnosis in (
            "GENERATOR_MISS",
            "PRUNING_MISS",
            "POSTERIOR_MISS",
            "EXECUTION_MISS",
            "NONE",
        )
    }
    conclusion = _diagnostic_conclusion(
        checkpoint_rows,
        winner,
        offline_confirmation,
    )
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": (
            "T8_6B_SELECTION_FAILED_CLOSED"
            if winner is None
            else (
                "READY_FOR_T8_6B_LIVE_CONFIRMATION"
                if bool((offline_confirmation or {}).get("passed"))
                else "T8_6B_REPAIR_V2_FAILED_CLOSED"
            )
        ),
        "manifest_checksum": manifest["manifest_checksum"],
        "parent_t8_6_scientific_checksum": manifest[
            "parent_t8_6_scientific_checksum"
        ],
        "inventory": inventory,
        "roots": len(episodes),
        "full_arm_predictions": len(full_arm_rows),
        "teacher_shock_roots": len(
            {row["episode_id"] for row in teacher_rows}
        ),
        "conditions": list(T8_6B_POLICIES),
        "selected_challenger": winner,
        "challenger_evaluations": challenger_evaluations,
        "offline_repair_v2_confirmation": offline_confirmation,
        "aggregates": v86._selection_aggregates(
            checkpoint_rows,
            update_rows,
            full_arm_rows,
        ),
        "failure_taxonomy": failure_taxonomy,
        "conclusion": conclusion,
        "elapsed_seconds": time.perf_counter() - started,
        "source_validation_authorized": False,
        "authority_authorized": False,
        "firewall": dict(manifest["firewall"]),
    }
    report["scientific_checksum"] = _checksum(
        {
            key: report[key]
            for key in (
                "manifest_checksum",
                "inventory",
                "selected_challenger",
                "challenger_evaluations",
                "offline_repair_v2_confirmation",
                "failure_taxonomy",
                "conclusion",
                "source_validation_authorized",
                "authority_authorized",
            )
        }
    )
    report["report_checksum"] = _checksum(report)
    _write_jsonl(destination / "checkpoint_rows.jsonl", checkpoint_rows)
    _write_jsonl(destination / "update_rows.jsonl", update_rows)
    _write_jsonl(destination / "full_arm_rows.jsonl", full_arm_rows)
    _write_jsonl(destination / "teacher_shock_rows.jsonl", teacher_rows)
    _write_jsonl(
        destination / "confirmation_checkpoint_rows.jsonl",
        confirmation_checkpoint_rows,
    )
    _write_jsonl(
        destination / "confirmation_update_rows.jsonl",
        confirmation_update_rows,
    )
    _write_jsonl(
        destination / "confirmation_full_arm_rows.jsonl",
        confirmation_full_arm_rows,
    )
    _write_jsonl(
        destination / "confirmation_teacher_shock_rows.jsonl",
        confirmation_teacher_rows,
    )
    _write_json(destination / "selection_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--shard-dir", default=str(v86.DEFAULT_SHARD_DIR))
    parser.add_argument("--schedules", default=str(DEFAULT_ACTION_SCHEDULES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--maximum-roots", type=int)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.freeze:
        result = freeze_manifest(
            output_path=args.manifest,
            schedules_path=args.schedules,
        )
    else:
        result = run_selection(
            manifest_path=args.manifest,
            shard_dir=args.shard_dir,
            schedules_path=args.schedules,
            output_dir=args.output_dir,
            maximum_roots=args.maximum_roots,
        )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0 if result.get("selected_challenger") else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHALLENGERS",
    "DEFAULT_ACTION_SCHEDULES",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "freeze_manifest",
    "load_manifest",
    "main",
    "run_selection",
    "select_challenger",
]
