"""SAGE.T8.0 paired live shadow pilot on source-train ARC-AGI-3 games.

The pilot executes the same deterministic legacy/controller trajectory twice:
once with SAGE.T off and once with SAGE.T in shadow mode.  It measures live
prediction calibration, surprise, entropy, teleological support, safety,
latency, repair activity and posterior stability without granting authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from theory.unified_cognition_ab_benchmark import _run_arm
from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)

from .controller import SageTConfig, SageTController
from .selection_autopsy import (
    load_frozen_manifest as load_t7_1_manifest,
)

FORMAT_VERSION = "sage-t8-live-shadow-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t8_0_frozen_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "live_shadow_pilot_v1"
TERMINAL_FAILURE_STATES = {"GAME_OVER", "FAILED", "FAILURE", "LOSE", "LOSS"}
WIN_STATES = {"WIN", "WON", "SUCCESS"}


@dataclass(frozen=True)
class LiveShadowRow:
    game_id: str
    seed: int
    reset_index: int
    step: int
    action_key: str
    action_matches_off: bool
    assessment_found: bool
    predicted_terminal: float | None
    predicted_goal: float | None
    predicted_progress: float | None
    actual_terminal: int
    actual_goal: int
    actual_progress: float
    surprise: float | None
    entropy_before: float | None
    entropy_after: float | None
    entropy_reduction: float | None
    particles_before: int
    particles_after: int
    effective_sample_size_after: float | None
    repairs_attempted_delta: int
    repairs_admitted_delta: int
    top_program_changed: bool | None
    top_probability_after: float | None
    decision_latency_ms: float | None
    observation_latency_ms: float | None


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
    encoded = _canonical(_json_safe(value)).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
        raise ValueError("SAGE.T8.0 manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported SAGE.T8.0 manifest")
    if payload.get("status") != "FROZEN_BEFORE_SOURCE_TRAIN_LIVE":
        raise ValueError("SAGE.T8.0 manifest is not frozen")
    base = load_t7_1_manifest()
    if payload.get("base_t7_1_manifest_checksum") != base["manifest_checksum"]:
        raise ValueError("SAGE.T8.0 base T7.1 manifest drifted")
    source_games = set(base["source_train_games"])
    configured = {
        str(game).split("-", 1)[0] for game in payload["source_train_games"]
    }
    forbidden = {
        str(game).split("-", 1)[0] for game in payload["forbidden_games"]
    }
    if not configured or not configured <= source_games:
        raise ValueError("SAGE.T8.0 contains a non-source-train game")
    if configured & forbidden:
        raise ValueError("SAGE.T8.0 source and forbidden games overlap")
    if payload["authority"]["mode"] != "shadow":
        raise ValueError("SAGE.T8.0 must remain shadow-only")
    expected_hash = payload.get("code_sha256", {}).get("live_shadow_pilot.py")
    if not expected_hash:
        raise ValueError("SAGE.T8.0 code hash is missing")
    if _file_sha256(Path(__file__)) != expected_hash:
        raise ValueError("SAGE.T8.0 live pilot code drifted")
    return payload


def runtime_capabilities() -> dict[str, Any]:
    try:
        import arc_agi
    except ImportError as error:
        return {
            "ready": False,
            "reason": f"arc_agi_import_error:{error}",
        }
    versions = {}
    for package in ("arc-agi", "arcengine"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    required = ("Arcade", "OperationMode", "EnvironmentWrapper")
    missing = [name for name in required if not hasattr(arc_agi, name)]
    return {
        "ready": not missing,
        "reason": "" if not missing else f"missing_sdk_symbols:{','.join(missing)}",
        "versions": versions,
        "arc_agi_path": str(getattr(arc_agi, "__file__", "")),
    }


class InstrumentedSageTController(SageTController):
    """Shadow controller with non-invasive timing and compact in-memory audit."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.decision_latencies_ms: list[float] = []
        self.observation_latencies_ms: list[float] = []
        self.compact_records: list[dict[str, Any]] = []

    def decide(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        try:
            return super().decide(**kwargs)
        finally:
            self.decision_latencies_ms.append(
                (time.perf_counter() - started) * 1000.0
            )

    def observe_transition(self, record: Any) -> None:
        started = time.perf_counter()
        try:
            super().observe_transition(record)
        finally:
            self.observation_latencies_ms.append(
                (time.perf_counter() - started) * 1000.0
            )

    def _record(self, record: Mapping[str, Any]) -> None:
        compact = _compact_audit_record(record)
        self.compact_records.append(compact)
        self._audit.append(compact)

    def summary(self) -> Mapping[str, Any]:
        payload = dict(super().summary())
        payload["instrumentation"] = {
            "decision_latencies_ms": tuple(self.decision_latencies_ms),
            "observation_latencies_ms": tuple(self.observation_latencies_ms),
            "records": tuple(self.compact_records),
        }
        return payload


def _compact_audit_record(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    for key in ("posterior_before", "posterior_after"):
        snapshot = payload.get(key)
        if isinstance(snapshot, Mapping):
            trimmed = dict(snapshot)
            trimmed["top"] = list(trimmed.get("top", ()))[:8]
            payload[key] = trimmed
    sequences = []
    for item in payload.get("sequences", ()) or ():
        compact = {
            key: value
            for key, value in dict(item).items()
            if key != "program_predictions"
        }
        predictions = list(dict(item).get("program_predictions", ()) or ())
        compact["prediction_particles"] = len(predictions)
        known = set()
        for prediction in predictions:
            for packet in prediction.get("packets", ()) or ():
                known.update(packet.get("known_channels", ()) or ())
        compact["known_channels"] = sorted(known)
        sequences.append(compact)
    if "sequences" in payload:
        payload["sequences"] = sequences
    return payload


def _controller_factory(
    *,
    mode: str,
    manifest: Mapping[str, Any],
) -> Any:
    caps = manifest["controller"]

    def factory(game_id: str) -> UnifiedCognitiveController:
        if mode == "off":
            return UnifiedCognitiveController(
                game_id,
                config=UnifiedCognitiveConfig(sage_t_authority_mode="off"),
            )
        sage_t = InstrumentedSageTController(
            config=SageTConfig(
                mode="shadow",
                maximum_programs=int(caps["maximum_programs"]),
                maximum_sequences=int(caps["maximum_sequences"]),
                maximum_particles_per_decision=int(
                    caps["maximum_particles_per_decision"]
                ),
                ordinary_horizon=int(caps["ordinary_horizon"]),
            )
        )
        return UnifiedCognitiveController(
            game_id,
            config=UnifiedCognitiveConfig(sage_t_authority_mode="shadow"),
            sage_t_controller=sage_t,
        )

    return factory


def _action_key(step: Mapping[str, Any]) -> str:
    name = str(step.get("action", "")).strip().upper()
    data = dict(step.get("action_args", {}) or {})
    return f"{name}:{_canonical(data)}"


def _assessment_for_action(
    decision: Mapping[str, Any],
    action_key: str,
) -> Mapping[str, Any] | None:
    matches = [
        item
        for item in decision.get("sequences", ()) or ()
        if item.get("sequence") and item["sequence"][0] == action_key
    ]
    if not matches:
        return None
    return min(matches, key=lambda item: len(item["sequence"]))


def _snapshot_value(
    snapshot: Mapping[str, Any],
    key: str,
) -> float | None:
    value = snapshot.get(key)
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _top_hash(snapshot: Mapping[str, Any]) -> str:
    top = list(snapshot.get("top", ()) or ())
    return "" if not top else str(top[0].get("program_hash", ""))


def _top_probability(snapshot: Mapping[str, Any]) -> float | None:
    top = list(snapshot.get("top", ()) or ())
    if not top:
        return None
    return _finite_float(top[0].get("probability"))


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def rows_from_paired_arms(
    *,
    game_id: str,
    seed: int,
    off: Mapping[str, Any],
    shadow: Mapping[str, Any],
) -> tuple[LiveShadowRow, ...]:
    sage = dict(
        shadow.get("controller_summary", {}).get(
            "sage_t_joint_program_posterior",
            {},
        )
        or {}
    )
    instrumentation = dict(sage.get("instrumentation", {}) or {})
    records = tuple(instrumentation.get("records", ()) or ())
    decisions = tuple(item for item in records if item.get("kind") == "decision")
    observations = tuple(
        item for item in records if item.get("kind") == "observation"
    )
    decision_latencies = tuple(
        float(value)
        for value in instrumentation.get("decision_latencies_ms", ()) or ()
    )
    observation_latencies = tuple(
        float(value)
        for value in instrumentation.get("observation_latencies_ms", ()) or ()
    )
    off_steps = [
        step
        for attempt in off.get("attempts", ()) or ()
        for step in attempt.get("trace", ()) or ()
    ]
    rows = []
    flat_index = 0
    for attempt in shadow.get("attempts", ()) or ():
        reset_index = int(attempt.get("reset_index", 0))
        for step in attempt.get("trace", ()) or ():
            decision = decisions[flat_index] if flat_index < len(decisions) else {}
            observation = (
                observations[flat_index] if flat_index < len(observations) else {}
            )
            before = dict(observation.get("posterior_before", {}) or {})
            after = dict(observation.get("posterior_after", {}) or {})
            action_key = _action_key(step)
            assessment = _assessment_for_action(decision, action_key)
            levels_delta = max(
                0,
                int(step.get("levels_after", 0))
                - int(step.get("levels_before", 0)),
            )
            state_after = str(step.get("game_state_after", "")).upper()
            actual_terminal = int(state_after in TERMINAL_FAILURE_STATES)
            actual_goal = int(levels_delta > 0 or state_after in WIN_STATES)
            entropy_before = _snapshot_value(before, "normalized_entropy")
            entropy_after = _snapshot_value(after, "normalized_entropy")
            off_key = (
                _action_key(off_steps[flat_index])
                if flat_index < len(off_steps)
                else ""
            )
            attempted_before = int(before.get("repairs_attempted", 0) or 0)
            attempted_after = int(after.get("repairs_attempted", 0) or 0)
            admitted_before = int(before.get("repairs_admitted", 0) or 0)
            admitted_after = int(after.get("repairs_admitted", 0) or 0)
            top_before = _top_hash(before)
            top_after = _top_hash(after)
            rows.append(
                LiveShadowRow(
                    game_id=str(game_id),
                    seed=int(seed),
                    reset_index=reset_index,
                    step=int(step.get("step", flat_index)),
                    action_key=action_key,
                    action_matches_off=action_key == off_key,
                    assessment_found=assessment is not None,
                    predicted_terminal=(
                        None
                        if assessment is None
                        else _finite_float(assessment.get("terminal_risk"))
                    ),
                    predicted_goal=(
                        None
                        if assessment is None
                        else _finite_float(assessment.get("expected_goal"))
                    ),
                    predicted_progress=(
                        None
                        if assessment is None
                        else _finite_float(assessment.get("expected_progress"))
                    ),
                    actual_terminal=actual_terminal,
                    actual_goal=actual_goal,
                    actual_progress=float(levels_delta),
                    surprise=_finite_float(observation.get("surprise")),
                    entropy_before=entropy_before,
                    entropy_after=entropy_after,
                    entropy_reduction=(
                        None
                        if entropy_before is None or entropy_after is None
                        else entropy_before - entropy_after
                    ),
                    particles_before=int(before.get("particles", 0) or 0),
                    particles_after=int(after.get("particles", 0) or 0),
                    effective_sample_size_after=_snapshot_value(
                        after,
                        "effective_sample_size",
                    ),
                    repairs_attempted_delta=max(
                        0,
                        attempted_after - attempted_before,
                    ),
                    repairs_admitted_delta=max(
                        0,
                        admitted_after - admitted_before,
                    ),
                    top_program_changed=(
                        None
                        if not top_before or not top_after
                        else top_before != top_after
                    ),
                    top_probability_after=_top_probability(after),
                    decision_latency_ms=(
                        decision_latencies[flat_index]
                        if flat_index < len(decision_latencies)
                        else None
                    ),
                    observation_latency_ms=(
                        observation_latencies[flat_index]
                        if flat_index < len(observation_latencies)
                        else None
                    ),
                )
            )
            flat_index += 1
    return tuple(rows)


def _percentile(values: Sequence[float], probability: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    position = (len(finite) - 1) * min(1.0, max(0.0, probability))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(finite),
        "mean": mean(finite) if finite else None,
        "median": median(finite) if finite else None,
        "p90": _percentile(finite, 0.90),
        "p95": _percentile(finite, 0.95),
        "maximum": max(finite) if finite else None,
    }


def _calibration(
    rows: Sequence[LiveShadowRow],
    *,
    prediction: str,
    observed: str,
) -> dict[str, Any]:
    pairs = [
        (float(getattr(row, prediction)), float(getattr(row, observed)))
        for row in rows
        if getattr(row, prediction) is not None
    ]
    bins = []
    for index in range(5):
        lower = index / 5.0
        upper = (index + 1) / 5.0
        selected = [
            (predicted, actual)
            for predicted, actual in pairs
            if lower <= predicted < upper or (index == 4 and predicted == 1.0)
        ]
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(selected),
                "mean_prediction": (
                    mean(value[0] for value in selected) if selected else None
                ),
                "empirical_rate": (
                    mean(value[1] for value in selected) if selected else None
                ),
            }
        )
    if not pairs:
        return {
            "n": 0,
            "positives": 0,
            "brier": None,
            "log_loss": None,
            "bins": bins,
        }
    epsilon = 1e-9
    return {
        "n": len(pairs),
        "positives": sum(int(actual > 0.0) for _, actual in pairs),
        "brier": mean((predicted - actual) ** 2 for predicted, actual in pairs),
        "log_loss": mean(
            -(
                actual * math.log(min(1.0 - epsilon, max(epsilon, predicted)))
                + (1.0 - actual)
                * math.log(
                    min(1.0 - epsilon, max(epsilon, 1.0 - predicted))
                )
            )
            for predicted, actual in pairs
        ),
        "bins": bins,
    }


def summarize_rows(rows: Sequence[LiveShadowRow]) -> dict[str, Any]:
    entropy = [
        float(row.entropy_reduction)
        for row in rows
        if row.entropy_reduction is not None
    ]
    surprise = [
        float(row.surprise) for row in rows if row.surprise is not None
    ]
    top_changes = [
        bool(row.top_program_changed)
        for row in rows
        if row.top_program_changed is not None
    ]
    progress_errors = [
        float(row.predicted_progress) - float(row.actual_progress)
        for row in rows
        if row.predicted_progress is not None
    ]
    return {
        "actions": len(rows),
        "prediction_coverage": (
            mean(row.assessment_found for row in rows) if rows else 0.0
        ),
        "calibration": {
            "terminal": _calibration(
                rows,
                prediction="predicted_terminal",
                observed="actual_terminal",
            ),
            "goal": _calibration(
                rows,
                prediction="predicted_goal",
                observed="actual_goal",
            ),
            "progress": {
                "n": len(progress_errors),
                "mae": (
                    mean(abs(value) for value in progress_errors)
                    if progress_errors
                    else None
                ),
                "rmse": (
                    math.sqrt(mean(value**2 for value in progress_errors))
                    if progress_errors
                    else None
                ),
            },
        },
        "surprise": _distribution(surprise),
        "entropy": {
            **_distribution(entropy),
            "positive_reduction_rate": (
                mean(value > 0.0 for value in entropy) if entropy else None
            ),
            "initial": rows[0].entropy_before if rows else None,
            "final": rows[-1].entropy_after if rows else None,
        },
        "teleology": {
            "progress_positive_actions": sum(
                row.actual_progress > 0.0 for row in rows
            ),
            "goal_positive_actions": sum(row.actual_goal for row in rows),
            "terminal_positive_actions": sum(row.actual_terminal for row in rows),
            "levels_completed": int(sum(row.actual_progress for row in rows)),
        },
        "latency_ms": {
            "decision": _distribution(
                [
                    float(row.decision_latency_ms)
                    for row in rows
                    if row.decision_latency_ms is not None
                ]
            ),
            "observation": _distribution(
                [
                    float(row.observation_latency_ms)
                    for row in rows
                    if row.observation_latency_ms is not None
                ]
            ),
        },
        "repairs": {
            "attempted": sum(row.repairs_attempted_delta for row in rows),
            "admitted": sum(row.repairs_admitted_delta for row in rows),
        },
        "stability": {
            "posterior_collapse_actions": sum(
                row.particles_after <= 0 for row in rows
            ),
            "top_program_churn_rate": (
                mean(top_changes) if top_changes else None
            ),
            "particle_count": _distribution(
                [float(row.particles_after) for row in rows]
            ),
            "effective_sample_size": _distribution(
                [
                    float(row.effective_sample_size_after)
                    for row in rows
                    if row.effective_sample_size_after is not None
                ]
            ),
            "top_probability": _distribution(
                [
                    float(row.top_probability_after)
                    for row in rows
                    if row.top_probability_after is not None
                ]
            ),
        },
    }


def build_report(
    rows: Sequence[LiveShadowRow],
    *,
    manifest: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    wall_clock_seconds: float,
) -> dict[str, Any]:
    metrics = summarize_rows(rows)
    per_game = {
        game: summarize_rows([row for row in rows if row.game_id == game])
        for game in sorted({row.game_id for row in rows})
    }
    safety = {
        "actions_match": (
            all(row.action_matches_off for row in rows)
            and all(bool(item.get("same_action_trace")) for item in conditions)
        ),
        "same_reset_states": all(
            bool(item.get("same_reset_states")) for item in conditions
        ),
        "controller_errors": sum(
            int(item.get("controller_errors", 0)) for item in conditions
        ),
        "illegal_actions": sum(
            int(item.get("illegal_actions", 0)) for item in conditions
        ),
        "environment_errors": sum(
            int(item.get("environment_errors", 0)) for item in conditions
        ),
        "interventions": sum(
            int(item.get("interventions", 0)) for item in conditions
        ),
        "trace_errors": sum(
            int(item.get("trace_errors", 0)) for item in conditions
        ),
    }
    gate = manifest["gate"]
    terminal_positives = int(
        metrics["teleology"]["terminal_positive_actions"]
    )
    goal_positives = int(metrics["teleology"]["goal_positive_actions"])
    latency_p95 = metrics["latency_ms"]["decision"]["p95"]
    checks = {
        "runtime_ready": bool(runtime.get("ready")),
        "minimum_actions": len(rows) >= int(gate["minimum_actions"]),
        "paired_action_identity": safety["actions_match"],
        "paired_reset_identity": safety["same_reset_states"],
        "zero_controller_errors": safety["controller_errors"] == 0,
        "zero_illegal_actions": safety["illegal_actions"] == 0,
        "zero_environment_errors": safety["environment_errors"] == 0,
        "zero_interventions": safety["interventions"] == 0,
        "zero_trace_errors": safety["trace_errors"] == 0,
        "prediction_coverage": (
            float(metrics["prediction_coverage"])
            >= float(gate["minimum_prediction_coverage"])
        ),
        "finite_surprise": (
            int(metrics["surprise"]["n"])
            >= int(gate["minimum_finite_surprise_samples"])
        ),
        "no_posterior_collapse": (
            int(metrics["stability"]["posterior_collapse_actions"]) == 0
        ),
        "decision_latency": (
            latency_p95 is not None
            and float(latency_p95) <= float(gate["maximum_decision_p95_ms"])
        ),
        "terminal_calibration_identified": (
            terminal_positives >= int(gate["minimum_terminal_positives"])
        ),
        "teleological_calibration_identified": (
            goal_positives >= int(gate["minimum_goal_positives"])
        ),
    }
    integration_names = (
        "runtime_ready",
        "minimum_actions",
        "paired_action_identity",
        "paired_reset_identity",
        "zero_controller_errors",
        "zero_illegal_actions",
        "zero_environment_errors",
        "zero_interventions",
        "zero_trace_errors",
        "prediction_coverage",
        "finite_surprise",
        "no_posterior_collapse",
        "decision_latency",
    )
    integration_passed = all(checks[name] for name in integration_names)
    calibration_identified = bool(
        checks["terminal_calibration_identified"]
        and checks["teleological_calibration_identified"]
    )
    if not integration_passed:
        diagnosis = "live_shadow_integration_or_latency_failure"
    elif not calibration_identified:
        diagnosis = "live_teleological_underidentification"
    else:
        diagnosis = "live_shadow_measurement_complete"
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": (
            "READY_FOR_EXTENDED_SOURCE_TRAIN_SHADOW"
            if integration_passed and calibration_identified
            else "DIAGNOSIS_COMPLETE_FAIL_CLOSED"
        ),
        "manifest_checksum": manifest["manifest_checksum"],
        "base_t7_1_manifest_checksum": manifest[
            "base_t7_1_manifest_checksum"
        ],
        "runtime": dict(runtime),
        "wall_clock_seconds": float(wall_clock_seconds),
        "conditions": [dict(item) for item in conditions],
        "rows": len(rows),
        "metrics": metrics,
        "per_game": per_game,
        "safety": safety,
        "checks": checks,
        "integration_gate_passed": integration_passed,
        "calibration_identified": calibration_identified,
        "diagnosis": diagnosis,
        "source_validation_authorized": False,
        "bounded_authority_authorized": False,
        "active_authority_authorized": False,
        "firewall": {
            "source_train_only": True,
            "source_validation_opened": False,
            "holdout_opened": False,
            "ar25_opened": False,
        },
    }
    payload["report_checksum"] = _checksum(payload)
    return payload


def _condition_summary(
    *,
    game_id: str,
    seed: int,
    off: Mapping[str, Any],
    shadow: Mapping[str, Any],
    wall_clock_seconds: float,
) -> dict[str, Any]:
    off_attempts = tuple(off.get("attempts", ()) or ())
    shadow_attempts = tuple(shadow.get("attempts", ()) or ())
    off_digests = tuple(item.get("reset_visual_digest") for item in off_attempts)
    shadow_digests = tuple(
        item.get("reset_visual_digest") for item in shadow_attempts
    )
    errors = tuple(shadow.get("controller_errors", ()) or ())
    off_actions = tuple(
        _action_key(step)
        for attempt in off_attempts
        for step in attempt.get("trace", ()) or ()
    )
    shadow_actions = tuple(
        _action_key(step)
        for attempt in shadow_attempts
        for step in attempt.get("trace", ()) or ()
    )
    environment_errors = sum(
        str(item.get("failure_cause", "")).startswith("environment_")
        for item in shadow_attempts
    )
    illegal = sum("unavailable_decision" in str(error) for error in errors)
    sage = dict(
        shadow.get("controller_summary", {}).get(
            "sage_t_joint_program_posterior",
            {},
        )
        or {}
    )
    return {
        "game_id": str(game_id),
        "seed": int(seed),
        "off_actions": int(off.get("actions_executed", 0) or 0),
        "shadow_actions": int(shadow.get("actions_executed", 0) or 0),
        "same_action_trace": off_actions == shadow_actions,
        "same_reset_states": off_digests == shadow_digests,
        "controller_errors": len(errors),
        "illegal_actions": illegal,
        "environment_errors": environment_errors,
        "interventions": int(sage.get("interventions", 0) or 0),
        "trace_errors": int(sage.get("trace_errors", 0) or 0),
        "effective_mode": str(sage.get("effective_mode", "")),
        "wall_clock_seconds": float(wall_clock_seconds),
    }


def run_live_shadow_pilot(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    environments_dir: str | Path = "environment_files",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    manifest = load_frozen_manifest(manifest_path)
    runtime = runtime_capabilities()
    expected_versions = {
        "arc-agi": str(manifest["runtime"]["arc_agi"]),
        "arcengine": str(manifest["runtime"]["arcengine"]),
    }
    observed_versions = dict(runtime.get("versions", {}) or {})
    versions_match = all(
        observed_versions.get(package) == version
        for package, version in expected_versions.items()
    )
    runtime["expected_versions"] = expected_versions
    runtime["versions_match"] = versions_match
    runtime["ready"] = bool(runtime.get("ready")) and versions_match
    if not versions_match and not runtime.get("reason"):
        runtime["reason"] = "sdk_version_mismatch"
    destination = Path(output_dir)
    if not runtime.get("ready"):
        payload = {
            "format_version": FORMAT_VERSION,
            "status": "BLOCKED_RUNTIME",
            "manifest_checksum": manifest["manifest_checksum"],
            "runtime": runtime,
            "source_validation_authorized": False,
            "active_authority_authorized": False,
        }
        payload["report_checksum"] = _checksum(payload)
        _write_json(destination / "report.json", payload)
        return payload
    rows: list[LiveShadowRow] = []
    conditions = []
    started = time.perf_counter()
    for game_id in manifest["source_train_games"]:
        for seed in manifest["seeds"]:
            condition_started = time.perf_counter()
            common = {
                "arm": "unified",
                "game_id": str(game_id),
                "seed": int(seed),
                "action_budget_per_reset": int(
                    manifest["action_budget_per_reset"]
                ),
                "resets": int(manifest["resets"]),
                "env_dir": Path(environments_dir),
                "env_factory": None,
            }
            off = _run_arm(
                controller_factory=_controller_factory(
                    mode="off",
                    manifest=manifest,
                ),
                **common,
            )
            shadow = _run_arm(
                controller_factory=_controller_factory(
                    mode="shadow",
                    manifest=manifest,
                ),
                **common,
            )
            rows.extend(
                rows_from_paired_arms(
                    game_id=str(game_id),
                    seed=int(seed),
                    off=off,
                    shadow=shadow,
                )
            )
            conditions.append(
                _condition_summary(
                    game_id=str(game_id),
                    seed=int(seed),
                    off=off,
                    shadow=shadow,
                    wall_clock_seconds=time.perf_counter() - condition_started,
                )
            )
    report = build_report(
        rows,
        manifest=manifest,
        conditions=conditions,
        runtime=runtime,
        wall_clock_seconds=time.perf_counter() - started,
    )
    _write_jsonl(destination / "rows.jsonl", tuple(asdict(row) for row in rows))
    _write_json(destination / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--check-runtime", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.check_runtime:
        result = runtime_capabilities()
    else:
        result = run_live_shadow_pilot(
            manifest_path=args.manifest,
            environments_dir=args.environments_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    if args.check_runtime:
        return 0 if result.get("ready") else 2
    return (
        0
        if result.get("status") == "READY_FOR_EXTENDED_SOURCE_TRAIN_SHADOW"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "InstrumentedSageTController",
    "LiveShadowRow",
    "build_report",
    "load_frozen_manifest",
    "main",
    "rows_from_paired_arms",
    "run_live_shadow_pilot",
    "runtime_capabilities",
    "summarize_rows",
]
