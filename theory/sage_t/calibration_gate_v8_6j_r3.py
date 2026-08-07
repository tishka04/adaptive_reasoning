"""T8.6j-r3 bounded-repair source-train gate."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

from theory.sage12.bound_mechanic_pilot import load_pairs

from . import calibration_gate_v8_6 as v86
from . import calibration_gate_v8_6c as v86c
from . import calibration_gate_v8_6i as v86i
from . import calibration_gate_v8_6j_r2 as r2
from . import live_shadow_pilot_v9 as live_r2
from .goal_generation_v3 import programs_for_with_structural_goal_guard
from .posterior_v8 import T8_6G_POLICIES
from .posterior_v11 import BudgetedRepairProgramPosterior
from .structural_roles import StructuralRoleProgramExecutor

FORMAT_VERSION = "sage-t8.6j-r3-bounded-repair-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t8_6j_r3_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "calibration_v8_6j_r3"
DEFAULT_PARENT_REPORT = (
    live_r2.DEFAULT_OUTPUT_DIR / "t8_6j_r2_long_live_report.json"
)
SELECTED_POLICY = r2.SELECTED_POLICY
MAXIMUM_REPAIR_CONTEXTS = 16


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: v86c._file_sha256(directory / name)
        for name in ("posterior_v11.py", "calibration_gate_v8_6j_r3.py")
    }


def _load_parent_report(path: str | Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T8.6j-r2 long-live report checksum mismatch")
    if report.get("status") != "T8_6J_R2_LONG_LIVE_FAILED_CLOSED":
        raise ValueError("T8.6j-r3 requires the failed r2 long-live gate")
    checks = dict(report.get("live_confirmation", {}).get("checks", {}))
    required_passes = set(checks) - {
        "four_hundred_actions",
        "per_game_observation_tail",
    }
    if not all(bool(checks.get(name)) for name in required_passes):
        raise ValueError("T8.6j-r2 failed outside horizon/assimilation")
    if checks.get("four_hundred_actions") is not False:
        raise ValueError("T8.6j-r3 requires the short realized horizon")
    if checks.get("per_game_observation_tail") is not False:
        raise ValueError("T8.6j-r3 requires the assimilation-tail failure")
    if report.get("source_validation_authorized") is not False:
        raise ValueError("source-validation opened after T8.6j-r2")
    return report


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    parent_report_path: str | Path = DEFAULT_PARENT_REPORT,
) -> dict[str, Any]:
    parent = _load_parent_report(parent_report_path)
    repair_gate = r2.load_manifest()
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T8_6J_R3_SOURCE_TRAIN_GATE",
        "frozen_at": "2026-08-06",
        "parent_live_report_checksum": parent["report_checksum"],
        "r2_manifest_checksum": repair_gate["manifest_checksum"],
        "code_sha256": _code_hashes(),
        "source_train_games": list(v86.EXPECTED_GAMES),
        "corpus": dict(v86.EXPECTED_CORPUS),
        "selected_policy": SELECTED_POLICY,
        "maximum_repair_contexts_per_game": MAXIMUM_REPAIR_CONTEXTS,
        "gates": {
            "offline_repair_v2_confirmation": True,
            "hidden_log_likelihood_nonnegative_by_game": True,
            "goal_teacher_shocks_preserved": True,
            "repair_budget_exercised": True,
        },
        "firewall": {
            "authority": "shadow",
            "source_train_only": True,
            "source_validation_opened": False,
            "holdout_opened": False,
            "ar25_opened": False,
        },
    }
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    return payload


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T8.6j-r3 manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T8.6j-r3 manifest")
    if payload.get("status") != "FROZEN_BEFORE_T8_6J_R3_SOURCE_TRAIN_GATE":
        raise ValueError("T8.6j-r3 manifest is not frozen")
    parent = _load_parent_report(DEFAULT_PARENT_REPORT)
    if payload.get("parent_live_report_checksum") != parent.get(
        "report_checksum"
    ):
        raise ValueError("T8.6j-r3 parent report drifted")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T8.6j-r3 code drifted")
    if payload.get("maximum_repair_contexts_per_game") != 16:
        raise ValueError("T8.6j-r3 repair budget drifted")
    if payload.get("firewall", {}).get("authority") != "shadow":
        raise ValueError("T8.6j-r3 firewall is open")
    return payload


def _new_posterior(
    policy,
    *,
    executor,
    manifest: Mapping[str, Any],
    dynamics_only: bool = False,
) -> BudgetedRepairProgramPosterior:
    config = manifest["posterior"]
    return BudgetedRepairProgramPosterior(
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
        maximum_repair_contexts=MAXIMUM_REPAIR_CONTEXTS,
    )


@contextmanager
def _runner() -> Iterable[None]:
    original_posterior = v86._new_posterior
    original_generator = v86._programs_for
    original_executor = v86.ProgramExecutor
    v86._new_posterior = _new_posterior
    v86._programs_for = programs_for_with_structural_goal_guard
    v86.ProgramExecutor = StructuralRoleProgramExecutor
    try:
        yield
    finally:
        v86._new_posterior = original_posterior
        v86._programs_for = original_generator
        v86.ProgramExecutor = original_executor


def _mean_by_game(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        game: mean(
            float(row["hidden_log_likelihood"])
            for row in rows
            if row["game"] == game
            and row.get("hidden_log_likelihood") is not None
        )
        for game in v86.EXPECTED_GAMES
    }


def _budget_probe(pairs, t7: Mapping[str, Any]) -> dict[str, Any]:
    evidence = []
    seen = set()
    for episode in v86._all_root_episodes(pairs):
        for panel in episode.panels:
            for arm in panel.arms:
                key = BudgetedRepairProgramPosterior._repair_context_key(arm)
                if key not in seen:
                    seen.add(key)
                    evidence.append(arm)
    first = evidence[0]
    while len(evidence) < MAXIMUM_REPAIR_CONTEXTS + 1:
        marker = f"budget_probe_{len(evidence)}"
        evidence.append(replace(first, events=(*first.events, marker)))
    actions = tuple(sorted({item.action.action_name for item in evidence}))
    programs = programs_for_with_structural_goal_guard(actions, (), t7)
    policy = T8_6G_POLICIES[SELECTED_POLICY].with_repair_v2()
    posterior = _new_posterior(
        policy,
        executor=StructuralRoleProgramExecutor(),
        manifest=t7,
    )
    posterior.seed(programs, initial_state=first.state_before)
    posterior.observe(first, allow_repair=False)
    for item in evidence[: MAXIMUM_REPAIR_CONTEXTS + 8]:
        posterior.repair(item)
    return dict(posterior.performance_snapshot())


def run_gate(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    shard_dir: str | Path = v86.DEFAULT_SHARD_DIR,
    schedules_path: str | Path = v86i.DEFAULT_ACTION_SCHEDULES,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    t7 = v86.load_t7_manifest(verify_code=True)
    pairs = load_pairs(str(shard_dir), v86.EXPECTED_GAMES)
    episodes = v86._all_root_episodes(pairs)
    schedules = json.loads(Path(schedules_path).read_text(encoding="utf-8"))
    policy = T8_6G_POLICIES[SELECTED_POLICY].with_repair_v2()
    baseline_rows = []
    candidate_rows = []
    confirmation_rows = []
    confirmation_updates = []
    confirmation_arms = []
    started = time.perf_counter()
    for episode in episodes:
        keys = [str(value) for value in schedules[episode.episode_id]]
        with r2._runner():
            baseline = v86._evaluate_episode(
                episode,
                policy=policy,
                manifest=t7,
                forced_keys=keys,
            )
        baseline_rows.extend(baseline[0])
        with _runner():
            candidate = v86._evaluate_episode(
                episode,
                policy=policy,
                manifest=t7,
                forced_keys=keys,
            )
            dynamics = v86._evaluate_episode(
                episode,
                policy=policy,
                manifest=t7,
                forced_keys=keys,
                dynamics_only=True,
            )
        candidate_rows.extend(candidate[0])
        confirmation_rows.extend((*candidate[0], *dynamics[0]))
        confirmation_updates.extend((*candidate[1], *dynamics[1]))
        confirmation_arms.extend((*candidate[3], *dynamics[3]))
    with _runner():
        teacher = v86._run_teacher_shocks(
            pairs,
            manifest=t7,
            policies={policy.name: policy},
        )
    offline = v86._offline_confirmation_checks(
        confirmation_rows,
        confirmation_updates,
        confirmation_arms,
        teacher,
        joint_condition=policy.name,
        dynamics_condition=f"{policy.name}_dynamics_only",
        manifest=v86i.load_manifest(),
    )
    baseline_ll = _mean_by_game(baseline_rows)
    candidate_ll = _mean_by_game(candidate_rows)
    deltas = {
        game: candidate_ll[game] - baseline_ll[game]
        for game in v86.EXPECTED_GAMES
    }
    goal_rows = [row for row in teacher if row["positive_kind"] == "goal"]
    budget = _budget_probe(pairs, t7)
    checks = {
        "offline_confirmation": bool(offline["passed"]),
        "hidden_log_likelihood_nonnegative": all(
            delta >= -1e-12 for delta in deltas.values()
        ),
        "goal_teacher_shocks": len(goal_rows) == 3
        and all(int(row["compatible_top8"]) > 0 for row in goal_rows),
        "repair_budget_exercised": int(budget["unique_repair_contexts"])
        == MAXIMUM_REPAIR_CONTEXTS
        and int(budget["repair_budget_skips"]) > 0,
        "source_validation_closed": True,
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": (
            "READY_FOR_T8_6J_R3_LONG_LIVE"
            if passed
            else "T8_6J_R3_FAILED_CLOSED"
        ),
        "manifest_checksum": manifest["manifest_checksum"],
        "checks": checks,
        "offline_confirmation": offline,
        "hidden_log_likelihood_delta_by_game": deltas,
        "goal_teacher_roots": len(goal_rows),
        "repair_budget_probe": budget,
        "elapsed_seconds": time.perf_counter() - started,
        "source_validation_authorized": False,
        "authority_authorized": False,
    }
    report["report_checksum"] = v86c._checksum(report)
    v86c._write_json(Path(output_dir) / "gate_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = (
        freeze_manifest(output_path=args.manifest)
        if args.freeze
        else run_gate(manifest_path=args.manifest, output_dir=args.output_dir)
    )
    print(json.dumps(v86c._json_safe(result), indent=2, sort_keys=True))
    return 0 if args.freeze or result.get("status") == "READY_FOR_T8_6J_R3_LONG_LIVE" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "MAXIMUM_REPAIR_CONTEXTS",
    "freeze_manifest",
    "load_manifest",
    "main",
    "run_gate",
]
