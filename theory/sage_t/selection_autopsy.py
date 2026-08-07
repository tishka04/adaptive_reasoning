"""SAGE.T7.1 posterior-selection autopsy with fair ablation scoring.

T7.1 is a source-train-only challenger layered on the immutable T7 baseline.
It never changes the baseline grammar, posterior, executor, coefficients, raw
rows, or gate artifacts.  Its purpose is to separate:

* a genuinely worse posterior;
* pruning/ranking of a good generated program;
* and an invalid ablation comparison caused by scoring different channels.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from theory.sage12.bound_mechanic_pilot import BindingPairRecord, load_pairs

from .contracts import ObservedTransition
from .decision import CounterfactualDecisionEngine
from .executor import ProgramExecutor
from .posterior import (
    DEFAULT_CHANNEL_WEIGHTS,
    ProgramPosterior,
    packet_log_likelihood,
)
from .replay_gate import (
    DEFAULT_MANIFEST_PATH as T7_MANIFEST_PATH,
)
from .replay_gate import (
    DEFAULT_V43_DIR,
    ReplayEpisode,
    _programs_for,
    episodes_from_binding_pairs,
    paired_bootstrap_interval,
)
from .replay_gate import load_frozen_manifest as load_t7_manifest
from .synthesis import AssembledProgram

FORMAT_VERSION = "sage-t7.1-selection-autopsy-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t7_1_frozen_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "selection_autopsy_v1"
CHANNELS = ("objects", "relations", "topology", "progress", "terminal", "goal")


@dataclass(frozen=True)
class SelectionAutopsyRow:
    episode_id: str
    source_game: str
    observations: int
    joint_native_log_likelihood: float
    dynamics_native_log_likelihood: float
    joint_common_log_likelihood: float
    dynamics_common_log_likelihood: float
    joint_common_minus_dynamics: float
    native_joint_minus_dynamics: float
    joint_channel_scores: Mapping[str, float]
    dynamics_channel_scores: Mapping[str, float]
    best_generated_log_likelihood: float
    joint_selection_regret: float
    best_exact_rank: int | None
    best_exact_probability: float
    best_family_rank: int | None
    best_family_probability_mass: float
    best_exact_pruned: bool
    best_family_pruned: bool
    top_prior_minus_best_prior: float | None
    top_evidence_minus_best_evidence: float | None
    generated_programs: int
    posterior_programs: int
    revealed_teleological_positives: int
    hidden_teleological_positives: int
    repairs_attempted: int
    repairs_admitted: int
    illegal_actions: int = 0
    execution_errors: int = 0


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
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
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


def load_frozen_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != _checksum(unsigned):
        raise ValueError("SAGE.T7.1 manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported SAGE.T7.1 manifest")
    if payload.get("status") != "FROZEN_BEFORE_SOURCE_AUTOPSY":
        raise ValueError("SAGE.T7.1 manifest is not frozen")
    base = load_t7_manifest(T7_MANIFEST_PATH)
    if payload.get("base_t7_manifest_checksum") != base["manifest_checksum"]:
        raise ValueError("SAGE.T7.1 base T7 manifest drifted")
    if payload["common_evaluation_weights"] != dict(DEFAULT_CHANNEL_WEIGHTS):
        raise ValueError("SAGE.T7.1 common scoring weights drifted")
    if payload["firewall"]["holdout_opened"] is not False:
        raise ValueError("SAGE.T7.1 holdout firewall is open")
    expected_code_hash = payload.get("code_sha256", {}).get(
        "selection_autopsy.py"
    )
    if not expected_code_hash:
        raise ValueError("SAGE.T7.1 code hash is missing")
    if _file_sha256(Path(__file__)) != expected_code_hash:
        raise ValueError("SAGE.T7.1 selection autopsy code drifted")
    return payload


def _posterior(
    *,
    base_manifest: Mapping[str, Any],
    executor: ProgramExecutor,
    condition: str,
) -> ProgramPosterior:
    config = base_manifest["posterior"]
    weights = dict(DEFAULT_CHANNEL_WEIGHTS)
    if condition == "dynamics_only":
        weights["progress"] = 0.0
        weights["goal"] = 0.0
    return ProgramPosterior(
        executor=executor,
        maximum_particles=int(config["maximum_particles"]),
        channel_weights=weights,
        unknown_coverage_penalty=float(config["unknown_coverage_penalty"]),
        repair_ess_threshold=float(config["repair_ess_threshold"]),
        repair_log_likelihood_threshold=float(
            config["repair_log_likelihood_threshold"]
        ),
    )


def _mixture_log_likelihood(
    posterior: ProgramPosterior,
    evidence: Sequence[ObservedTransition],
    *,
    executor: ProgramExecutor,
    score_weights: Mapping[str, float],
) -> float:
    values = []
    for arm in evidence:
        terms = []
        for particle in posterior.particles:
            prediction = executor.step(
                particle.program,
                arm.state_before,
                arm.action,
            )
            likelihood = packet_log_likelihood(
                prediction,
                arm.observation,
                channel_weights=score_weights,
                unknown_coverage_penalty=posterior.unknown_coverage_penalty,
            )
            terms.append(particle.log_weight + likelihood)
        if terms:
            values.append(_logsumexp(terms))
    return mean(values) if values else float("-inf")


def _channel_scores(
    posterior: ProgramPosterior,
    evidence: Sequence[ObservedTransition],
    *,
    executor: ProgramExecutor,
) -> dict[str, float]:
    return {
        channel: _mixture_log_likelihood(
            posterior,
            evidence,
            executor=executor,
            score_weights={
                name: float(name == channel)
                for name in CHANNELS
            },
        )
        for channel in CHANNELS
    }


def _program_score(
    program: AssembledProgram,
    evidence: Sequence[ObservedTransition],
    *,
    executor: ProgramExecutor,
    score_weights: Mapping[str, float],
) -> float:
    values = [
        packet_log_likelihood(
            executor.step(program.program, arm.state_before, arm.action),
            arm.observation,
            channel_weights=score_weights,
        )
        for arm in evidence
    ]
    return mean(values) if values else float("-inf")


def _teleological_positive(evidence: ObservedTransition) -> bool:
    packet = evidence.observation
    return bool(
        (
            packet.progress_mean is not None
            and abs(float(packet.progress_mean)) > 1e-12
        )
        or (
            packet.goal_probability is not None
            and float(packet.goal_probability) >= 0.5
        )
    )


def _ranking_diagnostics(
    posterior: ProgramPosterior,
    generated: Sequence[AssembledProgram],
    hidden: Sequence[ObservedTransition],
    *,
    executor: ProgramExecutor,
    common_weights: Mapping[str, float],
) -> dict[str, Any]:
    scored = [
        (
            _program_score(
                program,
                hidden,
                executor=executor,
                score_weights=common_weights,
            ),
            program,
        )
        for program in generated
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    ranked = sorted(
        posterior.particles,
        key=lambda particle: particle.log_weight,
        reverse=True,
    )
    exact_rank = next(
        (
            index
            for index, particle in enumerate(ranked, start=1)
            if particle.program.canonical_hash == best.program.canonical_hash
        ),
        None,
    )
    family = best.program.semantic_family
    family_ranks = [
        index
        for index, particle in enumerate(ranked, start=1)
        if particle.program.semantic_family == family
    ]
    family_mass = sum(
        particle.probability
        for particle in ranked
        if particle.program.semantic_family == family
    )
    exact_particle = next(
        (
            particle
            for particle in ranked
            if particle.program.canonical_hash == best.program.canonical_hash
        ),
        None,
    )
    top = ranked[0] if ranked else None
    prior_gap = None
    evidence_gap = None
    exact_probability = 0.0
    if exact_particle is not None and top is not None:
        exact_probability = exact_particle.probability
        prior_gap = top.log_prior - exact_particle.log_prior
        top_evidence = top.log_weight - top.log_prior
        best_evidence = exact_particle.log_weight - exact_particle.log_prior
        evidence_gap = top_evidence - best_evidence
    return {
        "best_score": best_score,
        "exact_rank": exact_rank,
        "exact_probability": exact_probability,
        "family_rank": min(family_ranks) if family_ranks else None,
        "family_mass": family_mass,
        "exact_pruned": exact_rank is None,
        "family_pruned": not family_ranks,
        "prior_gap": prior_gap,
        "evidence_gap": evidence_gap,
    }


def run_episode(
    episode: ReplayEpisode,
    *,
    base_manifest: Mapping[str, Any],
    autopsy_manifest: Mapping[str, Any],
    seed: int,
) -> tuple[SelectionAutopsyRow, ...]:
    executor = ProgramExecutor(
        maximum_cache_entries=int(
            base_manifest["executor"]["maximum_cache_entries"]
        )
    )
    actions = tuple(
        sorted(
            {
                arm.action.action_name
                for panel in episode.panels
                for arm in panel.arms
            }
        )
    )
    initial = _programs_for(actions, (), base_manifest)
    joint = _posterior(
        base_manifest=base_manifest,
        executor=executor,
        condition="joint",
    )
    dynamics = _posterior(
        base_manifest=base_manifest,
        executor=executor,
        condition="dynamics_only",
    )
    joint.seed(initial, initial_state=episode.panels[0].state)
    dynamics.seed(initial, initial_state=episode.panels[0].state)
    generated: dict[str, AssembledProgram] = {
        item.program.canonical_hash: item for item in initial
    }
    engine = CounterfactualDecisionEngine(
        executor=executor,
        maximum_sequences=int(base_manifest["decision"]["maximum_sequences"]),
        maximum_particles=int(base_manifest["decision"]["maximum_particles"]),
        ordinary_horizon=int(base_manifest["decision"]["ordinary_horizon"]),
    )
    checkpoints = {
        int(value) for value in autopsy_manifest["observation_checkpoints"]
    }
    common_weights = dict(autopsy_manifest["common_evaluation_weights"])
    revealed: list[ObservedTransition] = []
    hidden: list[ObservedTransition] = []
    rows = []
    rng = random.Random(seed)

    for index, panel in enumerate(episode.panels):
        available = {arm.action.key: arm for arm in panel.arms}
        decision = engine.decide(
            joint,
            panel.state,
            tuple(arm.action for arm in panel.arms),
        )
        selected_key = "" if decision.action is None else decision.action.key
        illegal = int(selected_key not in available)
        if illegal:
            selected_key = rng.choice(tuple(sorted(available)))
        chosen = available[selected_key]
        alternative = next(
            arm for key, arm in available.items() if key != selected_key
        )
        revealed.append(chosen)
        hidden.append(alternative)
        joint.observe(chosen)
        dynamics.observe(chosen)
        enriched = _programs_for(actions, revealed, base_manifest)
        for item in enriched:
            generated.setdefault(item.program.canonical_hash, item)
        joint.add_programs(enriched)
        dynamics.add_programs(enriched)
        for posterior in (joint, dynamics):
            for particle in posterior.particles:
                generated.setdefault(
                    particle.program.canonical_hash,
                    AssembledProgram(particle.program, particle.log_prior),
                )

        observations = index + 1
        if observations not in checkpoints:
            continue
        joint_native = _mixture_log_likelihood(
            joint,
            hidden,
            executor=executor,
            score_weights=joint.channel_weights,
        )
        dynamics_native = _mixture_log_likelihood(
            dynamics,
            hidden,
            executor=executor,
            score_weights=dynamics.channel_weights,
        )
        joint_common = _mixture_log_likelihood(
            joint,
            hidden,
            executor=executor,
            score_weights=common_weights,
        )
        dynamics_common = _mixture_log_likelihood(
            dynamics,
            hidden,
            executor=executor,
            score_weights=common_weights,
        )
        ranking = _ranking_diagnostics(
            joint,
            tuple(generated.values()),
            hidden,
            executor=executor,
            common_weights=common_weights,
        )
        snapshot = joint.snapshot(maximum_programs=0)
        rows.append(
            SelectionAutopsyRow(
                episode_id=episode.episode_id,
                source_game=episode.source_game,
                observations=observations,
                joint_native_log_likelihood=joint_native,
                dynamics_native_log_likelihood=dynamics_native,
                joint_common_log_likelihood=joint_common,
                dynamics_common_log_likelihood=dynamics_common,
                joint_common_minus_dynamics=joint_common - dynamics_common,
                native_joint_minus_dynamics=joint_native - dynamics_native,
                joint_channel_scores=_channel_scores(
                    joint,
                    hidden,
                    executor=executor,
                ),
                dynamics_channel_scores=_channel_scores(
                    dynamics,
                    hidden,
                    executor=executor,
                ),
                best_generated_log_likelihood=float(ranking["best_score"]),
                joint_selection_regret=max(
                    0.0,
                    float(ranking["best_score"]) - joint_common,
                ),
                best_exact_rank=ranking["exact_rank"],
                best_exact_probability=float(ranking["exact_probability"]),
                best_family_rank=ranking["family_rank"],
                best_family_probability_mass=float(ranking["family_mass"]),
                best_exact_pruned=bool(ranking["exact_pruned"]),
                best_family_pruned=bool(ranking["family_pruned"]),
                top_prior_minus_best_prior=ranking["prior_gap"],
                top_evidence_minus_best_evidence=ranking["evidence_gap"],
                generated_programs=len(generated),
                posterior_programs=len(joint.particles),
                revealed_teleological_positives=sum(
                    _teleological_positive(item) for item in revealed
                ),
                hidden_teleological_positives=sum(
                    _teleological_positive(item) for item in hidden
                ),
                repairs_attempted=int(snapshot["repairs_attempted"]),
                repairs_admitted=int(snapshot["repairs_admitted"]),
                illegal_actions=illegal,
            )
        )
    return tuple(rows)


def _run_game(
    payload: tuple[
        str,
        str,
        Mapping[str, Any],
        Mapping[str, Any],
    ],
) -> tuple[SelectionAutopsyRow, ...]:
    shard_dir, game, base_manifest, autopsy_manifest = payload
    pairs = load_pairs(shard_dir, (game,))
    episodes = episodes_from_binding_pairs(
        pairs,
        panels_per_episode=int(autopsy_manifest["panels_per_episode"]),
    )
    base_seed = int(autopsy_manifest["bootstrap"]["random_seed"])
    return tuple(
        row
        for index, episode in enumerate(episodes)
        for row in run_episode(
            episode,
            base_manifest=base_manifest,
            autopsy_manifest=autopsy_manifest,
            seed=base_seed + index,
        )
    )


def count_source_signals(pairs: Sequence[BindingPairRecord]) -> dict[str, int]:
    counts = {
        "arms": 0,
        "progress_positive": 0,
        "goal_positive": 0,
        "terminal_positive": 0,
    }
    for pair in pairs:
        for branch in (pair.left, pair.right):
            trace = branch.trace
            counts["arms"] += 1
            level = bool(trace.effects.level_complete) or (
                int(trace.levels_completed_after)
                > int(trace.levels_completed_before)
            )
            counts["progress_positive"] += int(level)
            counts["goal_positive"] += int(level)
            counts["terminal_positive"] += int(bool(trace.effects.game_over))
    return counts


def _paired_interval(
    rows: Sequence[SelectionAutopsyRow],
    *,
    left: str,
    right: str,
    checkpoint: int,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    selected = [row for row in rows if row.observations == checkpoint]
    interval = paired_bootstrap_interval(
        {row.episode_id: float(getattr(row, left)) for row in selected},
        {row.episode_id: float(getattr(row, right)) for row in selected},
        samples=int(manifest["bootstrap"]["samples"]),
        seed=int(manifest["bootstrap"]["random_seed"]) + checkpoint,
        confidence=float(manifest["bootstrap"]["confidence"]),
    )
    return asdict(interval)


def build_report(
    rows: Sequence[SelectionAutopsyRow],
    *,
    manifest: Mapping[str, Any],
    signal_counts: Mapping[str, int],
) -> dict[str, Any]:
    checkpoints = tuple(int(value) for value in manifest["observation_checkpoints"])
    comparisons = {}
    per_game = {}
    for checkpoint in checkpoints:
        selected = [row for row in rows if row.observations == checkpoint]
        common = _paired_interval(
            selected,
            left="joint_common_log_likelihood",
            right="dynamics_common_log_likelihood",
            checkpoint=checkpoint,
            manifest=manifest,
        )
        native = _paired_interval(
            selected,
            left="joint_native_log_likelihood",
            right="dynamics_native_log_likelihood",
            checkpoint=checkpoint,
            manifest=manifest,
        )
        comparisons[str(checkpoint)] = {
            "common_score_joint_minus_dynamics": common,
            "native_score_joint_minus_dynamics": native,
            "channel_omission_bias": {
                "mean": float(common["mean"]) - float(native["mean"]),
            },
        }
    for game in sorted({row.source_game for row in rows}):
        per_game[game] = {}
        for checkpoint in checkpoints:
            selected = [
                row
                for row in rows
                if row.source_game == game and row.observations == checkpoint
            ]
            per_game[game][str(checkpoint)] = {
                "episodes": len(selected),
                "common_joint_minus_dynamics": _mean_attribute(
                    selected,
                    "joint_common_minus_dynamics",
                ),
                "native_joint_minus_dynamics": _mean_attribute(
                    selected,
                    "native_joint_minus_dynamics",
                ),
                "selection_regret": _mean_attribute(
                    selected,
                    "joint_selection_regret",
                ),
                "exact_program_pruned_rate": mean(
                    row.best_exact_pruned for row in selected
                ),
                "family_pruned_rate": mean(
                    row.best_family_pruned for row in selected
                ),
                "mean_best_exact_rank": _mean_optional(
                    row.best_exact_rank for row in selected
                ),
                "mean_best_family_rank": _mean_optional(
                    row.best_family_rank for row in selected
                ),
                "mean_best_family_probability_mass": mean(
                    row.best_family_probability_mass for row in selected
                ),
                "mean_top_prior_gap": _mean_optional(
                    row.top_prior_minus_best_prior for row in selected
                ),
                "mean_top_evidence_gap": _mean_optional(
                    row.top_evidence_minus_best_evidence for row in selected
                ),
            }

    final = [row for row in rows if row.observations == max(checkpoints)]
    final_comparison = comparisons[str(max(checkpoints))]
    common_interval = final_comparison["common_score_joint_minus_dynamics"]
    native_interval = final_comparison["native_score_joint_minus_dynamics"]
    margin = float(manifest["gate"]["common_score_noninferiority_margin"])
    minimum_positives = int(
        manifest["gate"]["minimum_teleological_positive_arms"]
    )
    teleological_positives = min(
        int(signal_counts["progress_positive"]),
        int(signal_counts["goal_positive"]),
    )
    sufficient = teleological_positives >= minimum_positives
    common_noninferior = float(common_interval["lower"]) >= -margin
    native_rejects = float(native_interval["upper"]) < 0.0
    common_rejects = float(common_interval["upper"]) < -margin
    if native_rejects and common_noninferior and not sufficient:
        diagnosis = (
            "ablation_scoring_bias_plus_teleological_underidentification"
        )
    elif common_rejects:
        diagnosis = "posterior_selection"
    elif not sufficient:
        diagnosis = "teleological_underidentification"
    else:
        diagnosis = "no_selection_failure_detected"

    expected_penalty = float(manifest["audit"]["expected_progress_smoothing_penalty"])
    observed_native = float(native_interval["mean"])
    checks = {
        "all_expected_games_present": (
            {row.source_game for row in rows}
            == set(manifest["source_train_games"])
        ),
        "all_rows_finite": all(
            math.isfinite(row.joint_common_log_likelihood)
            and math.isfinite(row.dynamics_common_log_likelihood)
            for row in rows
        ),
        "zero_illegal_actions": sum(row.illegal_actions for row in rows) == 0,
        "zero_execution_errors": sum(row.execution_errors for row in rows) == 0,
        "common_score_noninferior": common_noninferior,
        "native_ablation_rejects_joint": native_rejects,
        "teleological_support_sufficient": sufficient,
    }
    validation_authorized = bool(
        checks["common_score_noninferior"]
        and checks["teleological_support_sufficient"]
        and checks["zero_illegal_actions"]
        and checks["zero_execution_errors"]
    )
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": (
            "READY_FOR_SOURCE_VALIDATION"
            if validation_authorized
            else "DIAGNOSIS_COMPLETE_FAIL_CLOSED"
        ),
        "manifest_checksum": manifest["manifest_checksum"],
        "base_t7_manifest_checksum": manifest["base_t7_manifest_checksum"],
        "episodes": len({row.episode_id for row in rows}),
        "rows": len(rows),
        "source_signal_counts": dict(signal_counts),
        "comparisons": comparisons,
        "per_game": per_game,
        "selection_summary_at_5": {
            "selection_regret": _mean_attribute(
                final,
                "joint_selection_regret",
            ),
            "exact_program_pruned_rate": mean(
                row.best_exact_pruned for row in final
            ),
            "family_pruned_rate": mean(
                row.best_family_pruned for row in final
            ),
            "mean_best_exact_rank": _mean_optional(
                row.best_exact_rank for row in final
            ),
            "mean_best_family_rank": _mean_optional(
                row.best_family_rank for row in final
            ),
            "mean_best_family_probability_mass": mean(
                row.best_family_probability_mass for row in final
            ),
            "mean_top_prior_gap": _mean_optional(
                row.top_prior_minus_best_prior for row in final
            ),
            "mean_top_evidence_gap": _mean_optional(
                row.top_evidence_minus_best_evidence for row in final
            ),
        },
        "scoring_bias_audit": {
            "expected_progress_smoothing_penalty": expected_penalty,
            "observed_native_joint_minus_dynamics": observed_native,
            "absolute_alignment_error": abs(observed_native - expected_penalty),
            "common_score_joint_minus_dynamics": float(common_interval["mean"]),
        },
        "checks": checks,
        "diagnosis": diagnosis,
        "source_validation_authorized": validation_authorized,
        "active_authority_authorized": False,
        "firewall": {
            "source_only": True,
            "source_validation_opened": False,
            "holdout_opened": False,
            "ar25_opened": False,
        },
    }
    payload["report_checksum"] = _checksum(payload)
    return payload


def _mean_attribute(rows: Sequence[SelectionAutopsyRow], name: str) -> float:
    return mean(float(getattr(row, name)) for row in rows)


def _mean_optional(values: Iterable[int | float | None]) -> float | None:
    items = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return mean(items) if items else None


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    maximum = max(values)
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def load_autopsy_rows(path: str | Path) -> tuple[SelectionAutopsyRow, ...]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            try:
                rows.append(SelectionAutopsyRow(**payload))
            except TypeError as error:
                raise ValueError(
                    f"invalid SAGE.T7.1 row at line {line_number}"
                ) from error
    return tuple(rows)


def _finalize_report(
    report: dict[str, Any],
    *,
    execution_workers: int,
) -> dict[str, Any]:
    report["execution_workers"] = int(execution_workers)
    unsigned = dict(report)
    unsigned.pop("report_checksum", None)
    report["report_checksum"] = _checksum(unsigned)
    return report


def rebuild_report_from_rows(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    v43_dir: str | Path = DEFAULT_V43_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    manifest = load_frozen_manifest(manifest_path)
    destination = Path(output_dir)
    rows = load_autopsy_rows(destination / "rows.jsonl")
    games = tuple(manifest["source_train_games"])
    pairs = load_pairs(Path(v43_dir) / "source_train_shards", games)
    report = build_report(
        rows,
        manifest=manifest,
        signal_counts=count_source_signals(pairs),
    )
    previous_path = destination / "report.json"
    execution_workers = 1
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        execution_workers = int(previous.get("execution_workers", 1))
    report = _finalize_report(
        report,
        execution_workers=execution_workers,
    )
    _write_json(previous_path, report)
    return report


def run_source_autopsy(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    v43_dir: str | Path = DEFAULT_V43_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    workers: int = 1,
) -> dict[str, Any]:
    manifest = load_frozen_manifest(manifest_path)
    base = load_t7_manifest(T7_MANIFEST_PATH)
    games = tuple(manifest["source_train_games"])
    shard_dir = Path(v43_dir) / "source_train_shards"
    missing = [
        str(shard_dir / f"{game}.jsonl")
        for game in games
        if not (shard_dir / f"{game}.jsonl").exists()
    ]
    if missing:
        payload = {
            "format_version": FORMAT_VERSION,
            "status": "BLOCKED_MISSING_SOURCE_TRAIN_SHARDS",
            "missing_shards": missing,
            "source_validation_authorized": False,
            "holdout_opened": False,
        }
        payload["report_checksum"] = _checksum(payload)
        _write_json(Path(output_dir) / "report.json", payload)
        return payload
    pairs = load_pairs(shard_dir, games)
    signals = count_source_signals(pairs)
    worker_count = max(1, int(workers))
    batches = [
        (str(shard_dir), game, base, manifest)
        for game in games
    ]
    if worker_count == 1:
        rows = tuple(
            row
            for batch in batches
            for row in _run_game(batch)
        )
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count
        ) as pool:
            rows = tuple(
                row
                for batch_rows in pool.map(_run_game, batches)
                for row in batch_rows
            )
    report = build_report(
        rows,
        manifest=manifest,
        signal_counts=signals,
    )
    report = _finalize_report(report, execution_workers=worker_count)
    destination = Path(output_dir)
    _write_jsonl(destination / "rows.jsonl", (asdict(row) for row in rows))
    _write_json(destination / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--v43-dir", default=str(DEFAULT_V43_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--rebuild-report",
        action="store_true",
        help="rebuild report.json from the immutable rows.jsonl",
    )
    args = parser.parse_args(argv)
    if args.rebuild_report:
        result = rebuild_report_from_rows(
            manifest_path=args.manifest,
            v43_dir=args.v43_dir,
            output_dir=args.output_dir,
        )
    else:
        result = run_source_autopsy(
            manifest_path=args.manifest,
            v43_dir=args.v43_dir,
            output_dir=args.output_dir,
            workers=args.workers,
        )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "diagnosis": result.get("diagnosis"),
                "checks": result.get("checks"),
                "source_validation_authorized": result.get(
                    "source_validation_authorized"
                ),
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.get("status") == "READY_FOR_SOURCE_VALIDATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "SelectionAutopsyRow",
    "build_report",
    "count_source_signals",
    "load_autopsy_rows",
    "load_frozen_manifest",
    "main",
    "rebuild_report_from_rows",
    "run_episode",
    "run_source_autopsy",
]
