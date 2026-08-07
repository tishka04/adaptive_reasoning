"""SAGE.T8.6 source-train calibration and generation/selection diagnosis.

The T7 posterior and every T7--T8.5 result remain immutable.  This runner
compares four generalized-Bayes update policies on a shared sequence of
materialized interventions.  Counterfactual arms are evaluator-only and never
enter proposal, update, repair, or action selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from theory.sage12.bound_mechanic_pilot import BindingPairRecord, load_pairs

from .contracts import ObservedTransition, PredictionPacket
from .decision import CounterfactualDecisionEngine
from .executor import ProgramExecutor
from .posterior import packet_log_likelihood
from .posterior_v2 import (
    T8_6_POLICIES,
    CalibratedProgramPosterior,
    PosteriorUpdatePolicy,
)
from .replay_gate import (
    ReplayEpisode,
    _discriminative_quality,
    _oracle_families,
    _programs_for,
    _weights,
    fast_panel_from_binding_pair,
    paired_bootstrap_interval,
)
from .replay_gate import load_frozen_manifest as load_t7_manifest
from .synthesis import AssembledProgram

FORMAT_VERSION = "sage-t8.6-calibrated-posterior-v1"
CONFIRMATION_FORMAT_VERSION = "sage-t8.6-confirmation-v1"
DEFAULT_SELECTION_MANIFEST = Path(__file__).with_name(
    "sage_t8_6_selection_manifest.json"
)
DEFAULT_CONFIRMATION_MANIFEST = Path(__file__).with_name(
    "sage_t8_6_confirmation_manifest.json"
)
DEFAULT_SHARD_DIR = (
    Path("training")
    / "sage12"
    / "bound_mechanic_pilot_v4_3"
    / "source_train_shards"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "calibration_v8_6"
EXPECTED_GAMES = ("lp85", "su15")
FORBIDDEN_GAMES = frozenset({"re86", "ls20", "sc25", "ar25"})
EXPECTED_CORPUS = {
    "pairs": 380,
    "arms": 760,
    "roots": 64,
    "natural_five_panel_roots": 52,
    "terminal_positive_arms": 115,
    "goal_positive_arms": 3,
    "terminal_positive_roots": 23,
    "goal_positive_roots": 3,
    "signal_roots": 25,
}
CHECKPOINTS = (1, 3, 5)


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


def _checkpoint_path(
    output_dir: Path,
    episode_id: str,
    condition: str,
) -> Path:
    digest = hashlib.sha256(
        f"{episode_id}:{condition}".encode()
    ).hexdigest()[:20]
    return output_dir / "checkpoints" / f"{digest}.json"


def _save_episode_checkpoint(
    path: Path,
    *,
    manifest_checksum: str,
    episode_id: str,
    condition: str,
    rows: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    arm_rows: Sequence[Mapping[str, Any]],
) -> None:
    payload: dict[str, Any] = {
        "manifest_checksum": manifest_checksum,
        "episode_id": episode_id,
        "condition": condition,
        "rows": list(rows),
        "updates": list(updates),
        "keys": list(keys),
        "arm_rows": list(arm_rows),
    }
    payload["checkpoint_checksum"] = _checksum(payload)
    _write_json(path, payload)


def _load_episode_checkpoint(
    path: Path,
    *,
    manifest_checksum: str,
    episode_id: str,
    condition: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    unsigned = dict(payload)
    checksum = str(unsigned.pop("checkpoint_checksum", ""))
    if checksum != _checksum(unsigned):
        return None
    if (
        payload.get("manifest_checksum") != manifest_checksum
        or payload.get("episode_id") != episode_id
        or payload.get("condition") != condition
    ):
        return None
    return (
        list(payload.get("rows", ())),
        list(payload.get("updates", ())),
        [str(value) for value in payload.get("keys", ())],
        list(payload.get("arm_rows", ())),
    )


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: _file_sha256(directory / name)
        for name in (
            "posterior_v2.py",
            "calibration_gate_v8_6.py",
            "live_shadow_pilot_v6.py",
        )
    }


def _data_hashes(shard_dir: Path) -> dict[str, str]:
    return {
        f"{game}.jsonl": _file_sha256(shard_dir / f"{game}.jsonl")
        for game in EXPECTED_GAMES
    }


def freeze_selection_manifest(
    *,
    output_path: str | Path = DEFAULT_SELECTION_MANIFEST,
    shard_dir: str | Path = DEFAULT_SHARD_DIR,
) -> dict[str, Any]:
    """Freeze code, data, coefficients, policies, seeds and fail-closed gates."""

    t7 = load_t7_manifest(verify_code=True)
    shards = Path(shard_dir)
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T8_6_SOURCE_TRAIN_SELECTION",
        "frozen_at": "2026-07-31",
        "base_t7_manifest_checksum": t7["manifest_checksum"],
        "code_sha256": _code_hashes(),
        "data_sha256": _data_hashes(shards),
        "source_train_games": list(EXPECTED_GAMES),
        "forbidden_games": sorted(FORBIDDEN_GAMES),
        "corpus": dict(EXPECTED_CORPUS),
        "checkpoints": list(CHECKPOINTS),
        "policies": {
            name: asdict(policy) for name, policy in T8_6_POLICIES.items()
        },
        "bootstrap": {"samples": 10_000, "seed": 8606, "confidence": 0.95},
        "selection": {
            "maximum_semantic_collapse_rate": 0.10,
            "minimum_collapse_reduction_fraction": 0.50,
            "paired_interval_lower_must_exceed": 0.0,
            "hidden_log_likelihood_game_delta_minimum": 0.0,
            "tie_breakers": [
                "terminal_log_loss",
                "hidden_log_likelihood",
                "latency",
            ],
        },
        "confirmation": {
            "minimum_prediction_coverage": 1.0,
            "maximum_false_terminal_rate_at_p80": 0.05,
            "minimum_terminal_compatible_top8_rate": 0.90,
            "minimum_goal_compatible_generated": 3,
            "minimum_goal_compatible_top8": 2,
            "minimum_goal_probability_gain": 0.25,
            "maximum_decision_p95_ms": 2500.0,
            "maximum_observation_p95_ms": 3000.0,
            "maximum_live_wall_seconds": 100.0,
            "maximum_latency_tail_ratio": 4.0,
            "maximum_repairs_per_observation": 1,
            "maximum_surviving_children_per_observation": 4,
        },
        "firewall": {
            "authority": "shadow",
            "source_train_only": True,
            "source_validation_opened": False,
            "holdout_opened": False,
            "ar25_opened": False,
            "llm_enabled": False,
            "neural_residual_weight": 0.0,
        },
    }
    payload["manifest_checksum"] = _checksum(payload)
    _write_json(Path(output_path), payload)
    return payload


def freeze_confirmation_manifest(
    selection_report: Mapping[str, Any],
    *,
    output_path: str | Path = DEFAULT_CONFIRMATION_MANIFEST,
) -> dict[str, Any]:
    """Freeze the selected challenger before any 50-action confirmation run."""

    winner = selection_report.get("selected_challenger")
    if not winner:
        raise ValueError("no T8.6 challenger passed selection")
    payload: dict[str, Any] = {
        "format_version": CONFIRMATION_FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T8_6_LIVE_CONFIRMATION",
        "selection_manifest_checksum": selection_report[
            "manifest_checksum"
        ],
        "selection_report_checksum": selection_report["report_checksum"],
        "selected_challenger": str(winner),
        "repair_policy": asdict(T8_6_POLICIES[str(winner)].with_repair_v2()),
        "code_sha256": _code_hashes(),
        "source_train_games": ["lp85-305b61c3", "su15-4c352900"],
        "actions": 50,
        "seeds": [0],
        "authority": "shadow",
        "source_validation_authorized": False,
    }
    payload["manifest_checksum"] = _checksum(payload)
    _write_json(Path(output_path), payload)
    return payload


def load_selection_manifest(
    path: str | Path = DEFAULT_SELECTION_MANIFEST,
    *,
    shard_dir: str | Path = DEFAULT_SHARD_DIR,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != _checksum(unsigned):
        raise ValueError("SAGE.T8.6 selection manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported SAGE.T8.6 selection manifest")
    if payload.get("status") != "FROZEN_BEFORE_T8_6_SOURCE_TRAIN_SELECTION":
        raise ValueError("SAGE.T8.6 selection manifest is not frozen")
    if tuple(payload.get("source_train_games", ())) != EXPECTED_GAMES:
        raise ValueError("SAGE.T8.6 game set drifted")
    if set(payload.get("source_train_games", ())) & FORBIDDEN_GAMES:
        raise ValueError("SAGE.T8.6 firewall violation")
    firewall = payload.get("firewall", {})
    if firewall.get("authority") != "shadow" or any(
        firewall.get(key) is not False
        for key in ("source_validation_opened", "holdout_opened", "ar25_opened")
    ):
        raise ValueError("SAGE.T8.6 firewall is open")
    if payload.get("corpus") != EXPECTED_CORPUS:
        raise ValueError("SAGE.T8.6 corpus contract drifted")
    if verify_hashes:
        if payload.get("code_sha256") != _code_hashes():
            raise ValueError("SAGE.T8.6 code drifted")
        if payload.get("data_sha256") != _data_hashes(Path(shard_dir)):
            raise ValueError("SAGE.T8.6 source-train data drifted")
    return payload


def corpus_inventory(pairs: Sequence[BindingPairRecord]) -> dict[str, int]:
    roots: dict[str, list[BindingPairRecord]] = defaultdict(list)
    terminal_roots: set[str] = set()
    goal_roots: set[str] = set()
    terminal_arms = 0
    goal_arms = 0
    for pair in pairs:
        if pair.source_split != "source_train" or pair.game_id not in EXPECTED_GAMES:
            raise ValueError("T8.6 read outside lp85/su15 source-train")
        roots[pair.root_key].append(pair)
        panel = fast_panel_from_binding_pair(pair)
        for arm in panel.arms:
            terminal = _positive(arm.observation.terminal_probability)
            goal = _positive(arm.observation.goal_probability)
            terminal_arms += int(terminal)
            goal_arms += int(goal)
            if terminal:
                terminal_roots.add(pair.root_key)
            if goal:
                goal_roots.add(pair.root_key)
    return {
        "pairs": len(pairs),
        "arms": 2 * len(pairs),
        "roots": len(roots),
        "natural_five_panel_roots": sum(len(items) >= 5 for items in roots.values()),
        "terminal_positive_arms": terminal_arms,
        "goal_positive_arms": goal_arms,
        "terminal_positive_roots": len(terminal_roots),
        "goal_positive_roots": len(goal_roots),
        "signal_roots": len(terminal_roots | goal_roots),
    }


def _positive(value: float | None) -> bool:
    return value is not None and float(value) >= 0.5


def _new_posterior(
    policy: PosteriorUpdatePolicy,
    *,
    executor: ProgramExecutor,
    manifest: Mapping[str, Any],
    dynamics_only: bool = False,
) -> CalibratedProgramPosterior:
    t7 = load_t7_manifest(verify_code=False)
    config = t7["posterior"]
    return CalibratedProgramPosterior(
        executor=executor,
        update_policy=policy,
        maximum_particles=int(config["maximum_particles"]),
        channel_weights=_weights("dynamics_only" if dynamics_only else "joint"),
        unknown_coverage_penalty=float(config["unknown_coverage_penalty"]),
        repair_ess_threshold=float(config["repair_ess_threshold"]),
        repair_log_likelihood_threshold=float(
            config["repair_log_likelihood_threshold"]
        ),
    )


def _mixture_prediction(
    posterior: CalibratedProgramPosterior,
    evidence: ObservedTransition,
) -> PredictionPacket:
    predictions = {
        particle.program.canonical_hash: posterior.executor.step(
            particle.program,
            evidence.state_before,
            evidence.action,
        )
        for particle in posterior.particles
    }
    return posterior.marginalize(predictions)


def _binary_metrics(probability: float | None, actual: int) -> tuple[float, float]:
    if probability is None:
        return float("nan"), float("nan")
    predicted = min(1.0 - 1e-12, max(1e-12, float(probability)))
    return (
        (predicted - int(actual)) ** 2,
        -(actual * math.log(predicted) + (1 - actual) * math.log(1 - predicted)),
    )


def _rank_and_mass(
    posterior: CalibratedProgramPosterior,
    families: frozenset[tuple[str, str]],
) -> tuple[int | None, float]:
    ranked = posterior.top(len(posterior.particles))
    matches = [
        (index, particle.probability)
        for index, particle in enumerate(ranked, start=1)
        if particle.program.semantic_family in families
    ]
    return (
        None if not matches else min(index for index, _ in matches),
        sum(probability for _, probability in matches),
    )


def _family_rank(
    programs: Sequence[Any],
    families: frozenset[tuple[str, str]],
) -> int | None:
    for index, item in enumerate(programs, start=1):
        program = getattr(item, "program", item)
        if program.semantic_family in families:
            return index
    return None


def _unpruned_programs(
    actions: Sequence[str],
    revealed: Sequence[ObservedTransition],
    manifest: Mapping[str, Any],
) -> tuple[Any, ...]:
    expanded = json.loads(json.dumps(manifest))
    expanded["generator"]["maximum_programs"] = 512
    expanded["generator"]["maximum_dynamics_beam"] = 64
    return _programs_for(actions, revealed, expanded)


def _evaluate_episode(
    episode: ReplayEpisode,
    *,
    policy: PosteriorUpdatePolicy,
    manifest: Mapping[str, Any],
    forced_keys: Sequence[str] = (),
    dynamics_only: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[dict[str, Any]],
]:
    executor = ProgramExecutor(
        maximum_cache_entries=int(manifest["executor"]["maximum_cache_entries"])
    )
    posterior = _new_posterior(
        policy,
        executor=executor,
        manifest=manifest,
        dynamics_only=dynamics_only,
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
    initial = _programs_for(actions, (), manifest)
    posterior.seed(initial, initial_state=episode.panels[0].state)
    generated = {item.program.canonical_hash: item for item in initial}
    engine = CounterfactualDecisionEngine(
        executor=executor,
        maximum_sequences=int(manifest["decision"]["maximum_sequences"]),
        maximum_particles=int(manifest["decision"]["maximum_particles"]),
        ordinary_horizon=int(manifest["decision"]["ordinary_horizon"]),
    )
    initial_entropy = posterior.normalized_entropy
    revealed: list[ObservedTransition] = []
    hidden: list[ObservedTransition] = []
    keys: list[str] = []
    checkpoint_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    full_arm_rows: list[dict[str, Any]] = []
    qualities: list[float] = []
    decision_latencies: list[float] = []
    for index, panel in enumerate(episode.panels[:5]):
        available = {arm.action.key: arm for arm in panel.arms}
        started = time.perf_counter()
        if index < len(forced_keys):
            selected_key = forced_keys[index]
        else:
            decision = engine.decide(
                posterior,
                panel.state,
                tuple(arm.action for arm in panel.arms),
            )
            selected_key = "" if decision.action is None else decision.action.key
        decision_ms = (time.perf_counter() - started) * 1000.0
        illegal = int(selected_key not in available)
        if illegal:
            selected_key = min(available)
        keys.append(selected_key)
        chosen = available[selected_key]
        alternative = next(arm for key, arm in available.items() if key != selected_key)
        qualities.append(_discriminative_quality(posterior, panel, selected_key, executor))
        decision_latencies.append(decision_ms)
        revealed.append(chosen)
        hidden.append(alternative)

        diagnostics = posterior.observe(chosen, allow_repair=False)
        update_rank_snapshot = tuple(posterior.particles)
        repair_before = posterior.snapshot(maximum_programs=0)
        if posterior._needs_repair():
            posterior.repair(chosen)
        repair_after = posterior.snapshot(maximum_programs=0)
        enriched = _programs_for(actions, revealed, manifest)
        for item in enriched:
            generated.setdefault(item.program.canonical_hash, item)
        posterior.add_programs(enriched)
        for particle in posterior.particles:
            generated.setdefault(
                particle.program.canonical_hash,
                AssembledProgram(particle.program, particle.log_prior),
            )
        update_rows.append(
            {
                "episode_id": episode.episode_id,
                "game": episode.source_game,
                "condition": policy.name + ("_dynamics_only" if dynamics_only else ""),
                "observation": index + 1,
                "action_key": selected_key,
                "illegal_action": illegal,
                "decision_latency_ms": decision_ms,
                "repair_cycles_delta": int(repair_after["repair_cycles"])
                - int(repair_before["repair_cycles"]),
                "repair_proposed_delta": int(repair_after["repairs_proposed"])
                - int(repair_before["repairs_proposed"]),
                "repair_evaluated_delta": int(repair_after["repairs_evaluated"])
                - int(repair_before["repairs_evaluated"]),
                "repair_survived_delta": int(repair_after["repairs_survived"])
                - int(repair_before["repairs_survived"]),
                **({} if diagnostics is None else diagnostics.to_dict()),
            }
        )
        checkpoint = index + 1
        if checkpoint not in CHECKPOINTS:
            continue
        oracle_families, _ = _oracle_families(
            revealed=revealed,
            hidden=hidden,
            available_actions=actions,
            manifest=manifest,
            executor=executor,
            generated_programs=tuple(generated.values()),
        )
        unpruned = _unpruned_programs(actions, revealed, manifest)
        production = _programs_for(actions, revealed, manifest)
        before_rank = _family_rank(unpruned, oracle_families)
        assembly_rank = _family_rank(production, oracle_families)
        update_rank = _family_rank(update_rank_snapshot, oracle_families)
        posterior_rank, posterior_mass = _rank_and_mass(posterior, oracle_families)
        oracle_family = next(iter(oracle_families), None)
        if before_rank is None:
            diagnosis = "GENERATOR_MISS"
        elif assembly_rank is None:
            diagnosis = "PRUNING_MISS"
        elif posterior_rank is None or posterior_rank > 8:
            diagnosis = "POSTERIOR_MISS"
        else:
            diagnosis = "NONE"
        arm_rows = []
        for arm in hidden:
            try:
                packet = _mixture_prediction(posterior, arm)
                likelihood = packet_log_likelihood(
                    packet,
                    arm.observation,
                    channel_weights=posterior.channel_weights,
                    unknown_coverage_penalty=posterior.unknown_coverage_penalty,
                )
                actual_terminal = int(_positive(arm.observation.terminal_probability))
                brier, log_loss = _binary_metrics(
                    packet.terminal_probability,
                    actual_terminal,
                )
                arm_rows.append(
                    {
                        "action_key": arm.action.key,
                        "log_likelihood": likelihood,
                        "terminal_probability": packet.terminal_probability,
                        "actual_terminal": actual_terminal,
                        "terminal_brier": brier,
                        "terminal_log_loss": log_loss,
                        "goal_probability": packet.goal_probability,
                        "actual_goal": int(_positive(arm.observation.goal_probability)),
                    }
                )
            except (ArithmeticError, KeyError, RuntimeError, TypeError, ValueError):
                diagnosis = "EXECUTION_MISS"
        checkpoint_rows.append(
            {
                "episode_id": episode.episode_id,
                "game": episode.source_game,
                "condition": policy.name + ("_dynamics_only" if dynamics_only else ""),
                "checkpoint": checkpoint,
                "hidden_arms": arm_rows,
                "hidden_log_likelihood": _finite_mean(
                    [item["log_likelihood"] for item in arm_rows]
                ),
                "terminal_brier": _finite_mean(
                    [item["terminal_brier"] for item in arm_rows]
                ),
                "terminal_log_loss": _finite_mean(
                    [item["terminal_log_loss"] for item in arm_rows]
                ),
                "entropy": posterior.normalized_entropy,
                "entropy_reduction": initial_entropy - posterior.normalized_entropy,
                "discriminative_action_quality": mean(qualities),
                "decision_latency_ms": mean(decision_latencies),
                "best_compatible_rank_before_pruning": before_rank,
                "compatible_rank_after_assembly": assembly_rank,
                "compatible_rank_after_update": update_rank,
                "compatible_rank_after_repair": posterior_rank,
                "compatible_mass_after_repair": posterior_mass,
                "dynamic_family": (
                    None if oracle_family is None else oracle_family[0]
                ),
                "goal_family": (
                    None if oracle_family is None else oracle_family[1]
                ),
                "diagnosis": diagnosis,
                "generated_programs": len(generated),
                "posterior_programs": len(posterior.particles),
            }
        )
    for panel_index, panel in enumerate(episode.panels):
        for arm in panel.arms:
            try:
                packet = _mixture_prediction(posterior, arm)
                likelihood = packet_log_likelihood(
                    packet,
                    arm.observation,
                    channel_weights=posterior.channel_weights,
                    unknown_coverage_penalty=posterior.unknown_coverage_penalty,
                )
                actual_terminal = int(
                    _positive(arm.observation.terminal_probability)
                )
                brier, log_loss = _binary_metrics(
                    packet.terminal_probability,
                    actual_terminal,
                )
                full_arm_rows.append(
                    {
                        "episode_id": episode.episode_id,
                        "panel_id": panel.panel_id,
                        "game": episode.source_game,
                        "condition": policy.name
                        + ("_dynamics_only" if dynamics_only else ""),
                        "panel_index": panel_index,
                        "action_key": arm.action.key,
                        "revealed_panel": panel_index < len(keys),
                        "revealed_arm": (
                            panel_index < len(keys)
                            and arm.action.key == keys[panel_index]
                        ),
                        "log_likelihood": likelihood,
                        "terminal_probability": packet.terminal_probability,
                        "actual_terminal": actual_terminal,
                        "terminal_brier": brier,
                        "terminal_log_loss": log_loss,
                        "goal_probability": packet.goal_probability,
                        "actual_goal": int(
                            _positive(arm.observation.goal_probability)
                        ),
                    }
                )
            except (ArithmeticError, KeyError, RuntimeError, TypeError, ValueError):
                full_arm_rows.append(
                    {
                        "episode_id": episode.episode_id,
                        "panel_id": panel.panel_id,
                        "game": episode.source_game,
                        "condition": policy.name
                        + ("_dynamics_only" if dynamics_only else ""),
                        "action_key": arm.action.key,
                        "execution_error": True,
                    }
                )
    return checkpoint_rows, update_rows, keys, full_arm_rows


def _all_root_episodes(
    pairs: Sequence[BindingPairRecord],
) -> tuple[ReplayEpisode, ...]:
    grouped: dict[str, list[BindingPairRecord]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.root_key].append(pair)
    return tuple(
        ReplayEpisode(
            episode_id=root_key,
            source_game=items[0].game_id,
            source_split=items[0].source_split,
            panels=tuple(
                fast_panel_from_binding_pair(pair)
                for pair in sorted(items, key=lambda item: (item.depth, item.path))
            ),
        )
        for root_key, items in sorted(grouped.items())
    )


def _progress_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if float(value) > 0.0:
        return "positive"
    if float(value) < 0.0:
        return "negative"
    return "zero"


def _compatible_prediction(
    prediction: PredictionPacket,
    observation: PredictionPacket,
) -> bool:
    if "progress" in observation.known_channels and (
        _progress_bucket(prediction.progress_mean)
        != _progress_bucket(observation.progress_mean)
    ):
        return False
    if "terminal" in observation.known_channels:
        if prediction.terminal_probability is None:
            return False
        if _positive(prediction.terminal_probability) != _positive(
            observation.terminal_probability
        ):
            return False
    if "goal" in observation.known_channels:
        if prediction.goal_probability is None:
            return False
        if _positive(prediction.goal_probability) != _positive(
            observation.goal_probability
        ):
            return False
    return True


def _compatible_programs(
    programs: Sequence[Any],
    signal: ObservedTransition,
    executor: ProgramExecutor,
) -> list[Any]:
    compatible = []
    for item in programs:
        program = getattr(item, "program", item)
        try:
            prediction = executor.step(
                program,
                signal.state_before,
                signal.action,
            )
        except (ArithmeticError, KeyError, RuntimeError, TypeError, ValueError):
            continue
        if _compatible_prediction(prediction, signal.observation):
            compatible.append(item)
    return compatible


def _signal_sequences(
    pairs: Sequence[BindingPairRecord],
) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, list[BindingPairRecord]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.root_key].append(pair)
    sequences = []
    for root_key, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: (item.depth, item.path))
        panels = [fast_panel_from_binding_pair(item) for item in ordered]
        signals = [
            (panel_index, arm)
            for panel_index, panel in enumerate(panels)
            for arm in panel.arms
            if _positive(arm.observation.terminal_probability)
            or _positive(arm.observation.goal_probability)
        ]
        if not signals:
            continue
        goal_signals = [
            item
            for item in signals
            if _positive(item[1].observation.goal_probability)
        ]
        positive_index, positive = (goal_signals or signals)[0]
        neutral = []
        for panel_index, panel in enumerate(panels):
            if panel_index == positive_index:
                continue
            candidate = next(
                (
                    arm
                    for arm in panel.arms
                    if not _positive(arm.observation.terminal_probability)
                    and not _positive(arm.observation.goal_probability)
                ),
                None,
            )
            if candidate is not None:
                neutral.append((panel, candidate))
            if len(neutral) >= 4:
                break
        sequence = [*neutral, (panels[positive_index], positive)]
        sequences.append(
            {
                "episode_id": root_key,
                "game": ordered[0].game_id,
                "panels": tuple(panel for panel, _ in sequence),
                "keys": tuple(arm.action.key for _, arm in sequence),
                "positive": positive,
                "positive_kind": (
                    "goal"
                    if _positive(positive.observation.goal_probability)
                    else "terminal"
                ),
            }
        )
    return tuple(sequences)


def _run_teacher_shocks(
    pairs: Sequence[BindingPairRecord],
    *,
    manifest: Mapping[str, Any],
    policies: Mapping[str, PosteriorUpdatePolicy] = T8_6_POLICIES,
) -> list[dict[str, Any]]:
    rows = []
    for sequence in _signal_sequences(pairs):
        episode = ReplayEpisode(
            episode_id=sequence["episode_id"],
            source_game=sequence["game"],
            source_split="source_train",
            panels=sequence["panels"],
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
        for name, policy in policies.items():
            executor = ProgramExecutor(
                maximum_cache_entries=int(
                    manifest["executor"]["maximum_cache_entries"]
                )
            )
            posterior = _new_posterior(
                policy,
                executor=executor,
                manifest=manifest,
            )
            posterior.seed(
                _programs_for(actions, (), manifest),
                initial_state=episode.panels[0].state,
            )
            revealed: list[ObservedTransition] = []
            pre_packet = PredictionPacket()
            positive_diagnostics = None
            for index, (panel, key) in enumerate(
                zip(episode.panels, sequence["keys"])
            ):
                evidence = next(arm for arm in panel.arms if arm.action.key == key)
                if index == len(episode.panels) - 1:
                    pre_packet = _mixture_prediction(posterior, evidence)
                revealed.append(evidence)
                positive_diagnostics = posterior.observe(
                    evidence,
                    allow_repair=False,
                )
                if posterior._needs_repair():
                    posterior.repair(evidence)
                posterior.add_programs(_programs_for(actions, revealed, manifest))
            positive = sequence["positive"]
            post_packet = _mixture_prediction(posterior, positive)
            unpruned = _unpruned_programs(actions, revealed, manifest)
            production = _programs_for(actions, revealed, manifest)
            top8 = posterior.top(8)
            unpruned_compatible = _compatible_programs(
                unpruned,
                positive,
                executor,
            )
            production_compatible = _compatible_programs(
                production,
                positive,
                executor,
            )
            top8_compatible = _compatible_programs(top8, positive, executor)
            if not unpruned_compatible:
                diagnosis = "GENERATOR_MISS"
            elif not production_compatible:
                diagnosis = "PRUNING_MISS"
            elif not top8_compatible:
                diagnosis = "POSTERIOR_MISS"
            else:
                diagnosis = "NONE"
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "game": episode.source_game,
                    "condition": name,
                    "positive_kind": sequence["positive_kind"],
                    "neutral_observations": len(episode.panels) - 1,
                    "terminal_probability_before": pre_packet.terminal_probability,
                    "terminal_probability_after": post_packet.terminal_probability,
                    "goal_probability_before": pre_packet.goal_probability,
                    "goal_probability_after": post_packet.goal_probability,
                    "goal_probability_gain": (
                        None
                        if pre_packet.goal_probability is None
                        or post_packet.goal_probability is None
                        else post_packet.goal_probability
                        - pre_packet.goal_probability
                    ),
                    "compatible_before_pruning": len(unpruned_compatible),
                    "compatible_after_assembly": len(production_compatible),
                    "compatible_top8": len(top8_compatible),
                    "semantic_collapse_after_positive": (
                        False
                        if positive_diagnostics is None
                        else positive_diagnostics.semantic_collapse
                    ),
                    "diagnosis": diagnosis,
                }
            )
    return rows


def _offline_confirmation_checks(
    checkpoint_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
    full_arm_rows: Sequence[Mapping[str, Any]],
    teacher_rows: Sequence[Mapping[str, Any]],
    *,
    joint_condition: str,
    dynamics_condition: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    config = manifest["confirmation"]
    joint_arms = [
        row
        for row in full_arm_rows
        if row.get("condition") == joint_condition
    ]
    valid_arms = [
        row for row in joint_arms if not row.get("execution_error")
    ]
    coverage = len(valid_arms) / max(1, len(joint_arms))
    false_terminal = [
        row
        for row in valid_arms
        if int(row["actual_terminal"]) == 0
        and row.get("terminal_probability") is not None
        and float(row["terminal_probability"]) >= 0.8
    ]
    negatives = [row for row in valid_arms if int(row["actual_terminal"]) == 0]
    false_terminal_rate = len(false_terminal) / max(1, len(negatives))
    teacher_joint = [
        row for row in teacher_rows if row["condition"] == joint_condition
    ]
    terminal_generated = [
        row
        for row in teacher_joint
        if row["positive_kind"] == "terminal"
        and int(row["compatible_after_assembly"]) > 0
    ]
    terminal_top8_rate = mean(
        float(int(row["compatible_top8"]) > 0) for row in terminal_generated
    ) if terminal_generated else 0.0
    goal_rows = [
        row for row in teacher_joint if row["positive_kind"] == "goal"
    ]
    goal_generated = sum(
        int(row["compatible_after_assembly"]) > 0 for row in goal_rows
    )
    goal_top8 = sum(int(row["compatible_top8"]) > 0 for row in goal_rows)
    goal_gains = [
        float(row["goal_probability_gain"])
        for row in goal_rows
        if row.get("goal_probability_gain") is not None
    ]
    goal_gain = _finite_mean(goal_gains)
    dynamics = _root_metric(
        checkpoint_rows,
        condition=dynamics_condition,
        metric="terminal_log_loss",
    )
    joint = _root_metric(
        checkpoint_rows,
        condition=joint_condition,
        metric="terminal_log_loss",
    )
    joint_advantage = paired_bootstrap_interval(
        dynamics,
        joint,
        samples=int(manifest["bootstrap"]["samples"]),
        seed=int(manifest["bootstrap"]["seed"]),
    )
    joint_updates = [
        row for row in update_rows if row["condition"] == joint_condition
    ]
    repairs_ok = all(
        int(row["repair_cycles_delta"])
        <= int(config["maximum_repairs_per_observation"])
        and int(row["repair_survived_delta"])
        <= int(config["maximum_surviving_children_per_observation"])
        for row in joint_updates
    )
    no_positive_collapse = not any(
        bool(row["semantic_collapse_after_positive"])
        for row in teacher_joint
    )
    checks = {
        "prediction_coverage": coverage
        >= float(config["minimum_prediction_coverage"]),
        "false_terminal_rate": false_terminal_rate
        <= float(config["maximum_false_terminal_rate_at_p80"]),
        "terminal_compatible_top8": terminal_top8_rate
        >= float(config["minimum_terminal_compatible_top8_rate"]),
        "goal_compatible_generated": goal_generated
        >= int(config["minimum_goal_compatible_generated"]),
        "goal_compatible_top8": goal_top8
        >= int(config["minimum_goal_compatible_top8"]),
        "goal_probability_gain": math.isfinite(goal_gain)
        and goal_gain >= float(config["minimum_goal_probability_gain"]),
        "joint_over_dynamics_only": joint_advantage.lower > 0.0,
        "repair_budgets": repairs_ok,
        "no_positive_semantic_collapse": no_positive_collapse,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": {
            "prediction_coverage": coverage,
            "false_terminal_rate_at_p80": false_terminal_rate,
            "terminal_compatible_top8_rate": terminal_top8_rate,
            "terminal_generated_roots": len(terminal_generated),
            "goal_compatible_generated": goal_generated,
            "goal_compatible_top8": goal_top8,
            "goal_probability_gain": goal_gain,
            "joint_over_dynamics_terminal_log_loss": asdict(joint_advantage),
        },
    }


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return mean(finite) if finite else float("nan")


def _run_or_resume_episode(
    episode: ReplayEpisode,
    *,
    policy: PosteriorUpdatePolicy,
    t7_manifest: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    output_dir: Path,
    forced_keys: Sequence[str] = (),
    dynamics_only: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[dict[str, Any]],
]:
    condition = policy.name + ("_dynamics_only" if dynamics_only else "")
    path = _checkpoint_path(output_dir, episode.episode_id, condition)
    resumed = _load_episode_checkpoint(
        path,
        manifest_checksum=str(selection_manifest["manifest_checksum"]),
        episode_id=episode.episode_id,
        condition=condition,
    )
    if resumed is not None:
        if forced_keys and resumed[2] != list(forced_keys):
            raise RuntimeError("resumed T8.6 action sequence drifted")
        return resumed
    result = _evaluate_episode(
        episode,
        policy=policy,
        manifest=t7_manifest,
        forced_keys=forced_keys,
        dynamics_only=dynamics_only,
    )
    _save_episode_checkpoint(
        path,
        manifest_checksum=str(selection_manifest["manifest_checksum"]),
        episode_id=episode.episode_id,
        condition=condition,
        rows=result[0],
        updates=result[1],
        keys=result[2],
        arm_rows=result[3],
    )
    return result


def _selection_aggregates(
    checkpoint_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
    full_arm_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_cell: dict[str, Any] = {}
    cells = sorted(
        {
            (str(row["game"]), str(row["condition"]), int(row["checkpoint"]))
            for row in checkpoint_rows
        }
    )
    for game, condition, checkpoint in cells:
        selected = [
            row
            for row in checkpoint_rows
            if row["game"] == game
            and row["condition"] == condition
            and int(row["checkpoint"]) == checkpoint
        ]
        by_cell[f"{game}:{condition}:{checkpoint}"] = {
            "roots": len(selected),
            "hidden_log_likelihood": _finite_mean(
                [float(row["hidden_log_likelihood"]) for row in selected]
            ),
            "terminal_brier": _finite_mean(
                [float(row["terminal_brier"]) for row in selected]
            ),
            "terminal_log_loss": _finite_mean(
                [float(row["terminal_log_loss"]) for row in selected]
            ),
            "entropy": _finite_mean(
                [float(row["entropy"]) for row in selected]
            ),
            "discriminative_action_quality": _finite_mean(
                [float(row["discriminative_action_quality"]) for row in selected]
            ),
        }
    channels: dict[str, Any] = {}
    for condition in sorted({str(row["condition"]) for row in full_arm_rows}):
        selected = [row for row in full_arm_rows if row["condition"] == condition]
        valid = [row for row in selected if not row.get("execution_error")]
        channels[condition] = {
            "arms": len(selected),
            "joint_log_likelihood": _finite_mean(
                [float(row["log_likelihood"]) for row in valid]
            ),
            "terminal": {
                "positives": sum(int(row["actual_terminal"]) for row in valid),
                "brier": _finite_mean(
                    [float(row["terminal_brier"]) for row in valid]
                ),
                "log_loss": _finite_mean(
                    [float(row["terminal_log_loss"]) for row in valid]
                ),
            },
            "goal": {
                "positives": sum(int(row["actual_goal"]) for row in valid),
                "prediction_coverage": mean(
                    float(row.get("goal_probability") is not None) for row in valid
                ) if valid else 0.0,
            },
        }
    latency_curve = {}
    for condition in sorted({str(row["condition"]) for row in update_rows}):
        values = []
        for observation in CHECKPOINTS:
            selected = [
                row
                for row in update_rows
                if row["condition"] == condition
                and int(row["observation"]) == observation
            ]
            values.append(
                {
                    "observation": observation,
                    "decision_ms": _finite_mean(
                        [float(row["decision_latency_ms"]) for row in selected]
                    ),
                    "update_ms": _finite_mean(
                        [float(row.get("elapsed_ms", math.nan)) for row in selected]
                    ),
                    "samples": len(selected),
                }
            )
        latency_curve[condition] = values
    return {
        "by_game_condition_checkpoint": by_cell,
        "by_channel": channels,
        "latency_curve": latency_curve,
    }


def _root_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    metric: str,
    checkpoint: int = 5,
    game: str | None = None,
) -> dict[str, float]:
    return {
        str(row["episode_id"]): float(row[metric])
        for row in rows
        if row["condition"] == condition
        and int(row["checkpoint"]) == checkpoint
        and (game is None or row["game"] == game)
        and math.isfinite(float(row[metric]))
    }


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
    evaluations: dict[str, Any] = {}
    survivors = []
    for name in ("tempered", "correlation_aware", "combined"):
        collapse = _collapse_rate(updates, name)
        metrics: dict[str, Any] = {
            "collapse_rate": collapse,
            "legacy_collapse_rate": legacy_collapse,
        }
        collapse_ok = collapse <= float(
            config["maximum_semantic_collapse_rate"]
        ) and (
            legacy_collapse <= 0.0
            or collapse <= legacy_collapse * (
                1.0 - float(config["minimum_collapse_reduction_fraction"])
            )
        )
        interval_checks = []
        for metric in ("terminal_brier", "terminal_log_loss"):
            legacy = _root_metric(rows, condition="legacy", metric=metric)
            challenger = _root_metric(rows, condition=name, metric=metric)
            interval = paired_bootstrap_interval(
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
        for game in EXPECTED_GAMES:
            legacy = _root_metric(
                rows,
                condition="legacy",
                metric="hidden_log_likelihood",
                game=game,
            )
            challenger = _root_metric(
                rows,
                condition=name,
                metric="hidden_log_likelihood",
                game=game,
            )
            shared = sorted(set(legacy) & set(challenger))
            delta = _finite_mean([challenger[key] - legacy[key] for key in shared])
            game_likelihood[game] = delta
        metrics["hidden_log_likelihood_delta_by_game"] = game_likelihood
        likelihood_ok = all(
            value >= float(config["hidden_log_likelihood_game_delta_minimum"])
            for value in game_likelihood.values()
        )
        metrics["checks"] = {
            "collapse": collapse_ok,
            "terminal_brier": interval_checks[0],
            "terminal_log_loss": interval_checks[1],
            "hidden_log_likelihood_both_games": likelihood_ok,
        }
        passed = all(metrics["checks"].values())
        metrics["passed"] = passed
        evaluations[name] = metrics
        if passed:
            terminal_loss = _finite_mean(
                list(
                    _root_metric(
                        rows,
                        condition=name,
                        metric="terminal_log_loss",
                    ).values()
                )
            )
            hidden_ll = _finite_mean(
                list(
                    _root_metric(
                        rows,
                        condition=name,
                        metric="hidden_log_likelihood",
                    ).values()
                )
            )
            latency = _finite_mean(
                [
                    float(row["decision_latency_ms"])
                    for row in rows
                    if row["condition"] == name and row["checkpoint"] == 5
                ]
            )
            survivors.append((terminal_loss, -hidden_ll, latency, name))
    winner = None if not survivors else min(survivors)[-1]
    return winner, evaluations


def _diagnostic_conclusion(
    rows: Sequence[Mapping[str, Any]],
    winner: str | None,
    updates: Sequence[Mapping[str, Any]],
) -> str:
    diagnoses = [
        str(row.get("diagnosis"))
        for row in rows
        if row.get("condition") == (winner or "legacy")
    ]
    if diagnoses.count("GENERATOR_MISS") > len(diagnoses) / 2:
        return "GENERATOR_LIMITED"
    if diagnoses.count("POSTERIOR_MISS") > len(diagnoses) / 2:
        return "SELECTION_LIMITED"
    if winner is None:
        return "INCONCLUSIVE_FAIL_CLOSED"
    repair_rows = [row for row in updates if row["condition"] == winner]
    if repair_rows and mean(float(row["repair_cycles_delta"]) for row in repair_rows) > 0.5:
        return "REPAIR_LIMITED"
    return "CALIBRATION_RECOVERED"


def run_selection(
    *,
    manifest_path: str | Path = DEFAULT_SELECTION_MANIFEST,
    shard_dir: str | Path = DEFAULT_SHARD_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    maximum_roots: int | None = None,
) -> dict[str, Any]:
    manifest = load_selection_manifest(manifest_path, shard_dir=shard_dir)
    t7 = load_t7_manifest(verify_code=True)
    pairs = load_pairs(str(shard_dir), EXPECTED_GAMES)
    inventory = corpus_inventory(pairs)
    if inventory != EXPECTED_CORPUS:
        raise ValueError(f"T8.6 corpus integrity failure: {inventory}")
    episodes = _all_root_episodes(pairs)
    if maximum_roots is not None:
        episodes = episodes[: max(0, int(maximum_roots))]
    checkpoint_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    full_arm_rows: list[dict[str, Any]] = []
    schedules: dict[str, list[str]] = {}
    destination = Path(output_dir)
    started = time.perf_counter()
    for episode in episodes:
        rows, updates, keys, arm_rows = _run_or_resume_episode(
            episode,
            policy=T8_6_POLICIES["legacy"],
            t7_manifest=t7,
            selection_manifest=manifest,
            output_dir=destination,
        )
        checkpoint_rows.extend(rows)
        update_rows.extend(updates)
        full_arm_rows.extend(arm_rows)
        schedules[episode.episode_id] = keys
        for name in ("tempered", "correlation_aware", "combined"):
            rows, updates, observed_keys, arm_rows = _run_or_resume_episode(
                episode,
                policy=T8_6_POLICIES[name],
                t7_manifest=t7,
                selection_manifest=manifest,
                output_dir=destination,
                forced_keys=keys,
            )
            if observed_keys != keys:
                raise RuntimeError("challenger action sequence drifted")
            checkpoint_rows.extend(rows)
            update_rows.extend(updates)
            full_arm_rows.extend(arm_rows)
    teacher_rows = _run_teacher_shocks(pairs, manifest=t7)
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
        repair_policy = T8_6_POLICIES[winner].with_repair_v2()
        for episode in episodes:
            keys = schedules[episode.episode_id]
            for dynamics_only in (False, True):
                rows, updates, observed_keys, arm_rows = _run_or_resume_episode(
                    episode,
                    policy=repair_policy,
                    t7_manifest=t7,
                    selection_manifest=manifest,
                    output_dir=destination,
                    forced_keys=keys,
                    dynamics_only=dynamics_only,
                )
                if observed_keys != keys:
                    raise RuntimeError("Repair V2 action sequence drifted")
                confirmation_checkpoint_rows.extend(rows)
                confirmation_update_rows.extend(updates)
                confirmation_full_arm_rows.extend(arm_rows)
        confirmation_teacher_rows = _run_teacher_shocks(
            pairs,
            manifest=t7,
            policies={repair_policy.name: repair_policy},
        )
        offline_confirmation = _offline_confirmation_checks(
            confirmation_checkpoint_rows,
            confirmation_update_rows,
            confirmation_full_arm_rows,
            confirmation_teacher_rows,
            joint_condition=repair_policy.name,
            dynamics_condition=f"{repair_policy.name}_dynamics_only",
            manifest=manifest,
        )
    live_negative_path = destination / "live_negative_report.json"
    live_negative_track: dict[str, Any]
    if live_negative_path.exists():
        candidate_live = json.loads(
            live_negative_path.read_text(encoding="utf-8")
        )
        unsigned_live = dict(candidate_live)
        live_checksum = str(unsigned_live.pop("report_checksum", ""))
        if (
            live_checksum == _checksum(unsigned_live)
            and candidate_live.get("manifest_checksum")
            == manifest["manifest_checksum"]
        ):
            live_negative_track = candidate_live
        else:
            live_negative_track = {"status": "INVALID_OR_STALE"}
    else:
        live_negative_track = {
            "status": (
                "NOT_RUN_OFFLINE_SELECTION_FAILED"
                if winner is None
                else "PENDING_BEFORE_CONFIRMATION"
            )
        }
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": (
            "CHALLENGER_SELECTED_PENDING_CONFIRMATION"
            if winner is not None
            else "T8_6_SELECTION_FAILED_CLOSED"
        ),
        "manifest_checksum": manifest["manifest_checksum"],
        "inventory": inventory,
        "roots": len(episodes),
        "natural_five_panel_roots": sum(
            len(episode.panels) >= 5 for episode in episodes
        ),
        "full_arm_predictions": len(full_arm_rows),
        "full_arm_predictions_per_condition": (
            0 if not T8_6_POLICIES else len(full_arm_rows) // len(T8_6_POLICIES)
        ),
        "teacher_shock_roots": len(
            {row["episode_id"] for row in teacher_rows}
        ),
        "conditions": list(T8_6_POLICIES),
        "checkpoints": list(CHECKPOINTS),
        "shared_action_schedules": len(schedules),
        "selected_challenger": winner,
        "challenger_evaluations": challenger_evaluations,
        "confirmation_gate": dict(manifest["confirmation"]),
        "offline_repair_v2_confirmation": offline_confirmation,
        "live_negative_track": live_negative_track,
        "aggregates": _selection_aggregates(
            checkpoint_rows,
            update_rows,
            full_arm_rows,
        ),
        "failure_taxonomy": {
            diagnosis: sum(row["diagnosis"] == diagnosis for row in checkpoint_rows)
            for diagnosis in (
                "GENERATOR_MISS",
                "PRUNING_MISS",
                "POSTERIOR_MISS",
                "EXECUTION_MISS",
                "NONE",
            )
        },
        "conclusion": _diagnostic_conclusion(
            checkpoint_rows,
            winner,
            update_rows,
        ),
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
    _write_json(destination / "action_schedules.json", schedules)
    if winner is not None and maximum_roots is None:
        freeze_confirmation_manifest(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_SELECTION_MANIFEST))
    parser.add_argument("--shard-dir", default=str(DEFAULT_SHARD_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--freeze-selection", action="store_true")
    parser.add_argument("--maximum-roots", type=int)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.freeze_selection:
        result = freeze_selection_manifest(
            output_path=args.manifest,
            shard_dir=args.shard_dir,
        )
    else:
        result = run_selection(
            manifest_path=args.manifest,
            shard_dir=args.shard_dir,
            output_dir=args.output_dir,
            maximum_roots=args.maximum_roots,
        )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0 if result.get("selected_challenger") else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIRMATION_FORMAT_VERSION",
    "DEFAULT_CONFIRMATION_MANIFEST",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SELECTION_MANIFEST",
    "EXPECTED_CORPUS",
    "FORMAT_VERSION",
    "corpus_inventory",
    "freeze_confirmation_manifest",
    "freeze_selection_manifest",
    "load_selection_manifest",
    "main",
    "run_selection",
    "select_challenger",
]
