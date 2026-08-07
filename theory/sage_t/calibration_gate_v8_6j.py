"""T8.6j exact incremental-posterior equivalence and performance gate."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from theory.sage12.bound_mechanic_pilot import load_pairs

from . import calibration_gate_v8_6 as v86
from . import calibration_gate_v8_6c as v86c
from . import calibration_gate_v8_6h as v86h
from . import calibration_gate_v8_6i as v86i
from . import live_shadow_pilot_v7 as live_i
from .goal_generation_v3 import programs_for_with_structural_goal_guard
from .posterior_v8 import T8_6G_POLICIES
from .posterior_v9 import IncrementalMinimumKLProgramPosterior
from .structural_roles import StructuralRoleProgramExecutor

FORMAT_VERSION = "sage-t8.6j-incremental-equivalence-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t8_6j_equivalence_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "calibration_v8_6j"
DEFAULT_PARENT_LIVE_REPORT = (
    live_i.DEFAULT_OUTPUT_DIR / "t8_6i_confirmation_report.json"
)
SELECTED_POLICY = v86i.SELECTED_POLICY
BENCHMARK_ACTIONS = 100


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: v86c._file_sha256(directory / name)
        for name in ("posterior_v9.py", "calibration_gate_v8_6j.py")
    }


def _load_parent_live_report(path: str | Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T8.6i live report checksum mismatch")
    if report.get("status") != "T8_6I_LIVE_CONFIRMATION_FAILED_CLOSED":
        raise ValueError("T8.6j requires the failed T8.6i live gate")
    checks = dict(report.get("live_confirmation", {}).get("checks", {}))
    failed = {name for name, passed in checks.items() if not bool(passed)}
    if failed != {"latency_tail_ratio"}:
        raise ValueError(f"T8.6i failure is not latency-only: {sorted(failed)}")
    if report.get("source_validation_authorized") is not False:
        raise ValueError("source-validation firewall opened after T8.6i")
    return report


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    parent_live_report_path: str | Path = DEFAULT_PARENT_LIVE_REPORT,
) -> dict[str, Any]:
    selection = v86i.load_manifest()
    selection_report = live_i._load_selection_report(
        live_i.DEFAULT_SELECTION_REPORT
    )
    live_report = _load_parent_live_report(parent_live_report_path)
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T8_6J_EQUIVALENCE",
        "frozen_at": "2026-08-06",
        "selection_manifest_checksum": selection["manifest_checksum"],
        "selection_scientific_checksum": selection_report[
            "scientific_checksum"
        ],
        "parent_live_report_checksum": live_report["report_checksum"],
        "parent_failure": "latency_tail_ratio_only",
        "code_sha256": _code_hashes(),
        "source_train_games": list(v86.EXPECTED_GAMES),
        "corpus": dict(v86.EXPECTED_CORPUS),
        "action_schedules_sha256": v86c._file_sha256(
            v86i.DEFAULT_ACTION_SCHEDULES
        ),
        "selected_policy": SELECTED_POLICY,
        "benchmark_actions": BENCHMARK_ACTIONS,
        "gates": {
            "exact_checkpoint_rows": True,
            "exact_update_science": True,
            "exact_hidden_arm_predictions": True,
            "exact_teacher_shocks": True,
            "exact_long_final_posterior": True,
            "minimum_benchmark_speedup": 2.0,
        },
        "frozen_invariants": {
            "likelihoods": "unchanged",
            "minimum_kl_projection": "unchanged",
            "repair_v2": "unchanged",
            "goal_generator": "unchanged_from_t8_6i",
            "structural_executor": "unchanged_from_t8_6i",
            "new_programs": "full_history_equivalent_replay",
            "unchanged_program_batch": "incremental_likelihood_cache",
            "semantic_signatures": "append_latest_transition_only",
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


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    parent_live_report_path: str | Path = DEFAULT_PARENT_LIVE_REPORT,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T8.6j manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T8.6j manifest")
    if payload.get("status") != "FROZEN_BEFORE_T8_6J_EQUIVALENCE":
        raise ValueError("T8.6j manifest is not frozen")
    live_report = _load_parent_live_report(parent_live_report_path)
    if payload.get("parent_live_report_checksum") != live_report.get(
        "report_checksum"
    ):
        raise ValueError("T8.6j parent live report drifted")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T8.6j code drifted")
    firewall = payload.get("firewall", {})
    if firewall.get("authority") != "shadow" or any(
        firewall.get(key) is not False
        for key in ("source_validation_opened", "holdout_opened", "ar25_opened")
    ):
        raise ValueError("T8.6j firewall is open")
    return payload


def _new_incremental_posterior(
    policy,
    *,
    executor,
    manifest: Mapping[str, Any],
    dynamics_only: bool = False,
) -> IncrementalMinimumKLProgramPosterior:
    config = manifest["posterior"]
    return IncrementalMinimumKLProgramPosterior(
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
def _incremental_runner() -> Iterable[None]:
    original_posterior = v86._new_posterior
    original_generator = v86._programs_for
    original_executor = v86.ProgramExecutor
    v86._new_posterior = _new_incremental_posterior
    v86._programs_for = programs_for_with_structural_goal_guard
    v86.ProgramExecutor = StructuralRoleProgramExecutor
    try:
        yield
    finally:
        v86._new_posterior = original_posterior
        v86._programs_for = original_generator
        v86.ProgramExecutor = original_executor


_NON_SCIENTIFIC_KEYS = frozenset(
    {
        "condition",
        "decision_latency_ms",
        "elapsed_ms",
        "executor_cache_hits_delta",
        "executor_cache_misses_delta",
    }
)


def _scientific(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _scientific(item)
            for key, item in value.items()
            if key not in _NON_SCIENTIFIC_KEYS
        }
    if isinstance(value, list):
        return [_scientific(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scientific(item) for item in value)
    return value


def _particle_fingerprint(posterior) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            particle.program.canonical_hash,
            particle.log_prior,
            particle.log_joint,
            particle.log_weight,
            particle.latest_log_likelihood,
            particle.latest_raw_log_likelihood,
            particle.observations,
            None
            if particle.state is None
            else particle.state.execution_signature,
        )
        for particle in posterior.particles
    )


def _long_benchmark(t7: Mapping[str, Any]) -> dict[str, Any]:
    pairs = load_pairs(str(v86.DEFAULT_SHARD_DIR), v86.EXPECTED_GAMES)
    sequence = next(iter(v86._signal_sequences(pairs)))
    evidence = next(
        arm
        for arm in sequence["panels"][0].arms
        if arm.action.key == sequence["keys"][0]
    )
    actions = (evidence.action.action_name,)
    programs = programs_for_with_structural_goal_guard(actions, (), t7)
    policy = T8_6G_POLICIES[SELECTED_POLICY].with_repair_v2()

    def execute(factory) -> tuple[float, Any, Mapping[str, Any]]:
        posterior = factory(
            policy,
            executor=StructuralRoleProgramExecutor(),
            manifest=t7,
        )
        posterior.seed(programs, initial_state=evidence.state_before)
        started = time.perf_counter()
        for _ in range(BENCHMARK_ACTIONS):
            posterior.add_programs(programs, initial_state=evidence.state_before)
            posterior.observe(evidence, allow_repair=False)
        elapsed = time.perf_counter() - started
        performance = (
            posterior.performance_snapshot()
            if hasattr(posterior, "performance_snapshot")
            else {}
        )
        return elapsed, posterior, performance

    baseline_seconds, baseline, _ = execute(v86h._new_minimum_kl_posterior)
    incremental_seconds, incremental, performance = execute(
        _new_incremental_posterior
    )
    return {
        "actions": BENCHMARK_ACTIONS,
        "baseline_seconds": baseline_seconds,
        "incremental_seconds": incremental_seconds,
        "speedup": baseline_seconds / max(incremental_seconds, 1e-12),
        "exact_final_posterior": (
            _particle_fingerprint(baseline)
            == _particle_fingerprint(incremental)
        ),
        "incremental_performance": dict(performance),
    }


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
    if v86.corpus_inventory(pairs) != v86.EXPECTED_CORPUS:
        raise ValueError("T8.6j corpus integrity failure")
    episodes = v86._all_root_episodes(pairs)
    schedules = json.loads(Path(schedules_path).read_text(encoding="utf-8"))
    policy = T8_6G_POLICIES[SELECTED_POLICY].with_repair_v2()
    mismatches = []
    started = time.perf_counter()
    for episode in episodes:
        keys = [str(value) for value in schedules[episode.episode_id]]
        with v86i._structural_goal_runner():
            baseline = v86._evaluate_episode(
                episode,
                policy=policy,
                manifest=t7,
                forced_keys=keys,
            )
        with _incremental_runner():
            incremental = v86._evaluate_episode(
                episode,
                policy=policy,
                manifest=t7,
                forced_keys=keys,
            )
        if _scientific(baseline) != _scientific(incremental):
            mismatches.append(episode.episode_id)
    with v86i._structural_goal_runner():
        baseline_teacher = v86._run_teacher_shocks(
            pairs,
            manifest=t7,
            policies={policy.name: policy},
        )
    with _incremental_runner():
        incremental_teacher = v86._run_teacher_shocks(
            pairs,
            manifest=t7,
            policies={policy.name: policy},
        )
    teacher_exact = _scientific(baseline_teacher) == _scientific(
        incremental_teacher
    )
    benchmark = _long_benchmark(t7)
    checks = {
        "all_64_roots_exact": not mismatches,
        "teacher_shocks_exact": teacher_exact,
        "long_final_posterior_exact": bool(
            benchmark["exact_final_posterior"]
        ),
        "minimum_speedup": float(benchmark["speedup"])
        >= float(manifest["gates"]["minimum_benchmark_speedup"]),
        "source_validation_closed": True,
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": (
            "READY_FOR_T8_6J_LONG_LIVE"
            if passed
            else "T8_6J_EQUIVALENCE_FAILED_CLOSED"
        ),
        "manifest_checksum": manifest["manifest_checksum"],
        "roots": len(episodes),
        "mismatched_roots": mismatches,
        "teacher_shocks": len(baseline_teacher),
        "benchmark": benchmark,
        "checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
        "source_validation_authorized": False,
        "authority_authorized": False,
    }
    report["report_checksum"] = v86c._checksum(report)
    v86c._write_json(Path(output_dir) / "equivalence_report.json", report)
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
    return 0 if args.freeze or result.get("status") == "READY_FOR_T8_6J_LONG_LIVE" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "freeze_manifest",
    "load_manifest",
    "main",
    "run_gate",
]
