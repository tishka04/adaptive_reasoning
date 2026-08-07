"""SAGE.T8.6i coordinate-free structural goal selection gate."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from theory.sage12.bound_mechanic_pilot import load_pairs

from . import calibration_gate_v8_6 as v86
from . import calibration_gate_v8_6c as v86c
from . import calibration_gate_v8_6h as v86h
from .goal_generation_v3 import programs_for_with_structural_goal_guard
from .posterior_v8 import T8_6G_POLICIES
from .structural_roles import (
    EASTMOST_TARGET,
    WESTMOST_TARGET,
    StructuralRoleProgramExecutor,
)

FORMAT_VERSION = "sage-t8.6i-structural-goal-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t8_6i_selection_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "calibration_v8_6i"
DEFAULT_PARENT_MANIFEST = v86h.DEFAULT_MANIFEST_PATH
DEFAULT_PARENT_REPORT = v86h.DEFAULT_OUTPUT_DIR / "selection_report.json"
DEFAULT_ACTION_SCHEDULES = v86h.DEFAULT_ACTION_SCHEDULES
SELECTED_POLICY = v86h.SELECTED_POLICY
CHALLENGERS = (SELECTED_POLICY,)
T8_6I_POLICIES = T8_6G_POLICIES


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: v86c._file_sha256(directory / name)
        for name in (
            "structural_roles.py",
            "goal_generation_v3.py",
            "calibration_gate_v8_6i.py",
        )
    }


def _load_parent_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("parent T8.6h report checksum mismatch")
    if payload.get("status") != "T8_6H_REPAIR_V2_FAILED_CLOSED":
        raise ValueError("T8.6i requires the failed T8.6h Repair V2 gate")
    if payload.get("selected_challenger") != SELECTED_POLICY:
        raise ValueError("T8.6h did not retain the frozen minimum-KL policy")
    if payload.get("conclusion") != "SELECTION_LIMITED":
        raise ValueError("T8.6i requires the T8.6h selection diagnosis")
    if payload.get("source_validation_authorized") is not False:
        raise ValueError("T8.6h source-validation firewall is open")
    return payload


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    parent_manifest_path: str | Path = DEFAULT_PARENT_MANIFEST,
    parent_report_path: str | Path = DEFAULT_PARENT_REPORT,
    schedules_path: str | Path = DEFAULT_ACTION_SCHEDULES,
) -> dict[str, Any]:
    parent_manifest = v86h.load_manifest(parent_manifest_path)
    parent_report = _load_parent_report(parent_report_path)
    schedule_file = Path(schedules_path)
    schedules = json.loads(schedule_file.read_text(encoding="utf-8"))
    if len(schedules) != int(parent_manifest["corpus"]["roots"]):
        raise ValueError("T8.6i schedules do not cover every root")
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T8_6I_SOURCE_TRAIN_SELECTION",
        "frozen_at": "2026-08-06",
        "parent_t8_6h_manifest_checksum": parent_manifest[
            "manifest_checksum"
        ],
        "parent_t8_6h_scientific_checksum": parent_report[
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
            for name, policy in T8_6I_POLICIES.items()
        },
        "bootstrap": dict(parent_manifest["bootstrap"]),
        "selection": dict(parent_manifest["selection"]),
        "confirmation": dict(parent_manifest["confirmation"]),
        "frozen_invariants": {
            "selected_posterior": SELECTED_POLICY,
            "posterior_implementation": "unchanged_from_t8_6g",
            "goal_bridge": "unchanged_from_t8_6h",
            "structural_roles": [WESTMOST_TARGET, EASTMOST_TARGET],
            "role_induction": "unique_horizontal_extrema_by_relative_order",
            "role_inputs": "target_role_and_relative_center_order_only",
            "forbidden_program_inputs": [
                "entity_ids",
                "raw_colors",
                "absolute_coordinates",
                "game_ids",
            ],
            "goal_guard": "observed_trigger_action_and_structural_role",
            "goal_guard_support": 0,
            "goal_guard_prior_delta": -0.10,
            "repair_requires_full_history_replay": True,
            "dsl": "unchanged",
            "mdl_priors": "unchanged",
            "channel_weights": "unchanged",
            "decision_utility": "unchanged",
            "actions": "exact_t8_6_schedules",
            "only_inference_change": (
                "coordinate_free_ordinal_roles_and_conditional_goal_guard"
            ),
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
        raise ValueError("SAGE.T8.6i manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported SAGE.T8.6i manifest")
    if payload.get("status") != "FROZEN_BEFORE_T8_6I_SOURCE_TRAIN_SELECTION":
        raise ValueError("SAGE.T8.6i manifest is not frozen")
    parent_manifest = v86h.load_manifest(parent_manifest_path)
    parent_report = _load_parent_report(parent_report_path)
    if payload.get("parent_t8_6h_manifest_checksum") != parent_manifest.get(
        "manifest_checksum"
    ):
        raise ValueError("T8.6i parent manifest drifted")
    if payload.get("parent_t8_6h_scientific_checksum") != parent_report.get(
        "scientific_checksum"
    ):
        raise ValueError("T8.6i parent scientific result drifted")
    if payload.get("corpus") != v86.EXPECTED_CORPUS:
        raise ValueError("T8.6i corpus contract drifted")
    firewall = payload.get("firewall", {})
    if firewall.get("authority") != "shadow" or any(
        firewall.get(key) is not False
        for key in ("source_validation_opened", "holdout_opened", "ar25_opened")
    ):
        raise ValueError("T8.6i firewall is open")
    if verify_hashes:
        if payload.get("code_sha256") != _code_hashes():
            raise ValueError("T8.6i code drifted")
        if payload.get("action_schedules_sha256") != v86c._file_sha256(
            Path(schedules_path)
        ):
            raise ValueError("T8.6i action schedules drifted")
    return payload


@contextmanager
def _structural_goal_runner() -> Iterable[None]:
    original_posterior = v86._new_posterior
    original_generator = v86._programs_for
    original_executor = v86.ProgramExecutor
    v86._new_posterior = v86h._new_minimum_kl_posterior
    v86._programs_for = programs_for_with_structural_goal_guard
    v86.ProgramExecutor = StructuralRoleProgramExecutor
    try:
        yield
    finally:
        v86._new_posterior = original_posterior
        v86._programs_for = original_generator
        v86.ProgramExecutor = original_executor


def _goal_selection_summary(
    teacher_rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in teacher_rows
        if row["condition"] == condition and row["positive_kind"] == "goal"
    ]
    return {
        "roots": len(selected),
        "compatible_generated": sum(
            int(row["compatible_after_assembly"]) > 0 for row in selected
        ),
        "compatible_top8": sum(
            int(row["compatible_top8"]) > 0 for row in selected
        ),
        "diagnoses": dict(
            sorted(Counter(str(row["diagnosis"]) for row in selected).items())
        ),
    }


def _diagnostic_conclusion(
    winner: str | None,
    offline_confirmation: Mapping[str, Any] | None,
) -> str:
    if winner is None:
        return "SELECTION_LIMITED"
    if not bool((offline_confirmation or {}).get("passed")):
        checks = dict((offline_confirmation or {}).get("checks", {}))
        if not checks.get("goal_compatible_generated", True):
            return "GENERATOR_LIMITED"
        if not checks.get("goal_compatible_top8", True):
            return "SELECTION_LIMITED"
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
        raise ValueError(f"T8.6i corpus integrity failure: {inventory}")
    episodes = v86._all_root_episodes(pairs)
    if maximum_roots is not None:
        episodes = episodes[: max(0, int(maximum_roots))]
    schedules = json.loads(Path(schedules_path).read_text(encoding="utf-8"))
    destination = Path(output_dir)
    checkpoint_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    full_arm_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with _structural_goal_runner():
        for episode in episodes:
            keys = [str(value) for value in schedules[episode.episode_id]]
            for policy in T8_6I_POLICIES.values():
                rows, updates, observed_keys, arms = v86._run_or_resume_episode(
                    episode,
                    policy=policy,
                    t7_manifest=t7,
                    selection_manifest=manifest,
                    output_dir=destination,
                    forced_keys=keys,
                )
                if observed_keys != keys:
                    raise RuntimeError("T8.6i action sequence drifted")
                checkpoint_rows.extend(rows)
                update_rows.extend(updates)
                full_arm_rows.extend(arms)
        teacher_rows = v86._run_teacher_shocks(
            pairs,
            manifest=t7,
            policies=T8_6I_POLICIES,
        )
    winner, challenger_evaluations = v86h.select_challenger(
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
        repair_policy = T8_6I_POLICIES[winner].with_repair_v2()
        with _structural_goal_runner():
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
                        raise RuntimeError("T8.6i Repair V2 schedule drifted")
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
    repair_condition = f"{SELECTED_POLICY}_repair_v2"
    goal_summary = _goal_selection_summary(
        confirmation_teacher_rows,
        condition=repair_condition,
    )
    conclusion = _diagnostic_conclusion(winner, offline_confirmation)
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": (
            "T8_6I_SELECTION_FAILED_CLOSED"
            if winner is None
            else (
                "READY_FOR_T8_6I_LIVE_CONFIRMATION"
                if bool((offline_confirmation or {}).get("passed"))
                else "T8_6I_REPAIR_V2_FAILED_CLOSED"
            )
        ),
        "manifest_checksum": manifest["manifest_checksum"],
        "parent_t8_6h_scientific_checksum": manifest[
            "parent_t8_6h_scientific_checksum"
        ],
        "inventory": inventory,
        "roots": len(episodes),
        "full_arm_predictions": len(full_arm_rows),
        "teacher_shock_roots": len(
            {row["episode_id"] for row in teacher_rows}
        ),
        "conditions": list(T8_6I_POLICIES),
        "selected_challenger": winner,
        "challenger_evaluations": challenger_evaluations,
        "offline_repair_v2_confirmation": offline_confirmation,
        "aggregates": v86._selection_aggregates(
            checkpoint_rows,
            update_rows,
            full_arm_rows,
        ),
        "minimum_kl_projection": v86h._projection_summary(update_rows),
        "structural_goal_selection": goal_summary,
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
                "minimum_kl_projection",
                "structural_goal_selection",
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
    return 0 if args.freeze or result.get("selected_challenger") else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHALLENGERS",
    "DEFAULT_ACTION_SCHEDULES",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "SELECTED_POLICY",
    "freeze_manifest",
    "load_manifest",
    "main",
    "run_selection",
]
