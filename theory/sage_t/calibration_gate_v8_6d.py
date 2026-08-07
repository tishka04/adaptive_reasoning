"""SAGE.T8.6d terminal-only likelihood-temperature sweep."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from theory.sage12.bound_mechanic_pilot import load_pairs

from . import calibration_gate_v8_6 as v86
from . import calibration_gate_v8_6b as v86b
from . import calibration_gate_v8_6c as v86c
from .executor import ProgramExecutor
from .posterior_v3 import (
    ChannelCalibratedProgramPosterior,
    ChannelPosteriorUpdatePolicy,
)
from .posterior_v5 import T8_6D_POLICIES

FORMAT_VERSION = "sage-t8.6d-terminal-temperature-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t8_6d_selection_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "calibration_v8_6d"
DEFAULT_PARENT_MANIFEST = v86c.DEFAULT_MANIFEST_PATH
DEFAULT_PARENT_REPORT = v86c.DEFAULT_OUTPUT_DIR / "selection_report.json"
DEFAULT_ACTION_SCHEDULES = v86c.DEFAULT_ACTION_SCHEDULES
CHALLENGERS = (
    "terminal_tempered_20",
    "terminal_tempered_15",
    "terminal_tempered_10",
)


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: v86c._file_sha256(directory / name)
        for name in ("posterior_v5.py", "calibration_gate_v8_6d.py")
    }


def _load_parent_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("parent T8.6c report checksum mismatch")
    if payload.get("status") != "T8_6C_SELECTION_FAILED_CLOSED":
        raise ValueError("T8.6d requires the failed T8.6c selection")
    if payload.get("selected_challenger") is not None:
        raise ValueError("T8.6c already selected a challenger")
    if payload.get("source_validation_authorized") is not False:
        raise ValueError("T8.6c source-validation firewall is open")
    return payload


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    parent_manifest_path: str | Path = DEFAULT_PARENT_MANIFEST,
    parent_report_path: str | Path = DEFAULT_PARENT_REPORT,
    schedules_path: str | Path = DEFAULT_ACTION_SCHEDULES,
) -> dict[str, Any]:
    parent_manifest = v86c.load_manifest(parent_manifest_path)
    parent_report = _load_parent_report(parent_report_path)
    schedule_file = Path(schedules_path)
    schedules = json.loads(schedule_file.read_text(encoding="utf-8"))
    if len(schedules) != int(parent_manifest["corpus"]["roots"]):
        raise ValueError("T8.6d schedules do not cover every root")
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T8_6D_SOURCE_TRAIN_SELECTION",
        "frozen_at": "2026-08-06",
        "parent_t8_6c_manifest_checksum": parent_manifest[
            "manifest_checksum"
        ],
        "parent_t8_6c_scientific_checksum": parent_report[
            "scientific_checksum"
        ],
        "action_schedules_sha256": v86c._file_sha256(schedule_file),
        "code_sha256": _code_hashes(),
        "source_train_games": list(v86.EXPECTED_GAMES),
        "forbidden_games": sorted(v86.FORBIDDEN_GAMES),
        "corpus": dict(v86.EXPECTED_CORPUS),
        "checkpoints": list(v86.CHECKPOINTS),
        "policies": {
            name: v86c._json_safe(asdict(policy))
            for name, policy in T8_6D_POLICIES.items()
        },
        "bootstrap": dict(parent_manifest["bootstrap"]),
        "selection": dict(parent_manifest["selection"]),
        "confirmation": dict(parent_manifest["confirmation"]),
        "frozen_invariants": {
            "terminal_temperatures": [0.20, 0.15, 0.10],
            "baseline_terminal_temperature": 0.25,
            "other_channel_temperatures": 1.0,
            "repeated_context_discount": False,
            "family_projection": False,
            "dsl": "unchanged",
            "executor": "unchanged",
            "mdl_priors": "unchanged",
            "channel_weights": "unchanged",
            "decision_utility": "unchanged",
            "actions": "exact_t8_6_schedules",
            "only_inference_change": "fixed_terminal_temperature",
        },
        "firewall": dict(parent_manifest["firewall"]),
    }
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
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
    if checksum != v86c._checksum(unsigned):
        raise ValueError("SAGE.T8.6d manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported SAGE.T8.6d manifest")
    if payload.get("status") != "FROZEN_BEFORE_T8_6D_SOURCE_TRAIN_SELECTION":
        raise ValueError("SAGE.T8.6d manifest is not frozen")
    parent_manifest = v86c.load_manifest(parent_manifest_path)
    parent_report = _load_parent_report(parent_report_path)
    if payload.get("parent_t8_6c_manifest_checksum") != parent_manifest.get(
        "manifest_checksum"
    ):
        raise ValueError("T8.6d parent manifest drifted")
    if payload.get("parent_t8_6c_scientific_checksum") != parent_report.get(
        "scientific_checksum"
    ):
        raise ValueError("T8.6d parent scientific result drifted")
    if payload.get("corpus") != v86.EXPECTED_CORPUS:
        raise ValueError("T8.6d corpus contract drifted")
    firewall = payload.get("firewall", {})
    if firewall.get("authority") != "shadow" or any(
        firewall.get(key) is not False
        for key in ("source_validation_opened", "holdout_opened", "ar25_opened")
    ):
        raise ValueError("T8.6d firewall is open")
    if verify_hashes:
        if payload.get("code_sha256") != _code_hashes():
            raise ValueError("T8.6d code drifted")
        if payload.get("action_schedules_sha256") != v86c._file_sha256(
            Path(schedules_path)
        ):
            raise ValueError("T8.6d action schedules drifted")
    return payload


def _new_temperature_posterior(
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
def _temperature_runner() -> Iterable[None]:
    original = v86._new_posterior
    v86._new_posterior = _new_temperature_posterior
    try:
        yield
    finally:
        v86._new_posterior = original


def select_challenger(
    rows: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    original = v86b.CHALLENGERS
    v86b.CHALLENGERS = CHALLENGERS
    try:
        return v86b.select_challenger(rows, updates, manifest=manifest)
    finally:
        v86b.CHALLENGERS = original


def _diagnostic_conclusion(
    rows: Sequence[Mapping[str, Any]],
    winner: str | None,
    offline_confirmation: Mapping[str, Any] | None,
) -> str:
    selected = [
        str(row.get("diagnosis"))
        for row in rows
        if row.get("condition") == (winner or "terminal_tempered_25")
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
        raise ValueError(f"T8.6d corpus integrity failure: {inventory}")
    episodes = v86._all_root_episodes(pairs)
    if maximum_roots is not None:
        episodes = episodes[: max(0, int(maximum_roots))]
    schedules = json.loads(Path(schedules_path).read_text(encoding="utf-8"))
    destination = Path(output_dir)
    checkpoint_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    full_arm_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with _temperature_runner():
        for episode in episodes:
            keys = [str(value) for value in schedules[episode.episode_id]]
            for policy in T8_6D_POLICIES.values():
                rows, updates, observed_keys, arms = v86._run_or_resume_episode(
                    episode,
                    policy=policy,
                    t7_manifest=t7,
                    selection_manifest=manifest,
                    output_dir=destination,
                    forced_keys=keys,
                )
                if observed_keys != keys:
                    raise RuntimeError("T8.6d action sequence drifted")
                checkpoint_rows.extend(rows)
                update_rows.extend(updates)
                full_arm_rows.extend(arms)
        teacher_rows = v86._run_teacher_shocks(
            pairs,
            manifest=t7,
            policies=T8_6D_POLICIES,
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
        repair_policy = T8_6D_POLICIES[winner].with_repair_v2()
        with _temperature_runner():
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
                        raise RuntimeError("T8.6d Repair V2 schedule drifted")
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
            "T8_6D_SELECTION_FAILED_CLOSED"
            if winner is None
            else (
                "READY_FOR_T8_6D_LIVE_CONFIRMATION"
                if bool((offline_confirmation or {}).get("passed"))
                else "T8_6D_REPAIR_V2_FAILED_CLOSED"
            )
        ),
        "manifest_checksum": manifest["manifest_checksum"],
        "parent_t8_6c_scientific_checksum": manifest[
            "parent_t8_6c_scientific_checksum"
        ],
        "inventory": inventory,
        "roots": len(episodes),
        "full_arm_predictions": len(full_arm_rows),
        "teacher_shock_roots": len(
            {row["episode_id"] for row in teacher_rows}
        ),
        "conditions": list(T8_6D_POLICIES),
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
    report["scientific_checksum"] = v86c._checksum(
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
    report["report_checksum"] = v86c._checksum(report)
    v86c._write_jsonl(destination / "checkpoint_rows.jsonl", checkpoint_rows)
    v86c._write_jsonl(destination / "update_rows.jsonl", update_rows)
    v86c._write_jsonl(destination / "full_arm_rows.jsonl", full_arm_rows)
    v86c._write_jsonl(destination / "teacher_shock_rows.jsonl", teacher_rows)
    v86c._write_jsonl(
        destination / "confirmation_checkpoint_rows.jsonl",
        confirmation_checkpoint_rows,
    )
    v86c._write_jsonl(
        destination / "confirmation_update_rows.jsonl",
        confirmation_update_rows,
    )
    v86c._write_jsonl(
        destination / "confirmation_full_arm_rows.jsonl",
        confirmation_full_arm_rows,
    )
    v86c._write_jsonl(
        destination / "confirmation_teacher_shock_rows.jsonl",
        confirmation_teacher_rows,
    )
    v86c._write_json(destination / "selection_report.json", report)
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
    print(json.dumps(v86c._json_safe(result), indent=2, sort_keys=True))
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
