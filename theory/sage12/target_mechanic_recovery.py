"""Runtime-safe recovery replication for SAGE12 target mechanics (V4.2.1)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from theory.sage11.splits import SOURCE_TRAIN, SOURCE_VALIDATION

from . import mechanic_induction as v4
from . import mechanic_replication as v41
from . import target_mechanic_replication as v42

PREFLIGHT_FORMAT_VERSION = "sage12-target-mechanic-preflight-v4.2.1"
RESULT_FORMAT_VERSION = "sage12-target-mechanic-pilot-result-v4.2.1"
REHEARSAL_FORMAT_VERSION = "sage12-target-mechanic-rehearsal-v4.2.1"
FAILURE_FORMAT_VERSION = "sage12-target-mechanic-runtime-failure-v4.2.1"
DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "mechanic_induction_v4_2_1"
DEFAULT_FROZEN_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "frozen_manifest.json"
V3_OUTPUT_DIR = Path("training") / "sage12" / "action_target_pilot_v3"
STATE_ANCHOR_CONDITIONS = v42.COARSE_ANCHOR_CONDITIONS
RULE_ANCHOR_CONDITIONS = (*STATE_ANCHOR_CONDITIONS, "any")
TARGET_EFFECT_LABELS = v42.TARGET_EFFECT_LABELS
MODEL_MODES = v42.MODEL_MODES
BASELINE_MODES = v42.BASELINE_MODES
compact_qwen_prompt = v42.compact_qwen_prompt
compact_qwen_schema = v42.compact_qwen_schema


def public_rule(rule: v4.MechanicRule) -> dict[str, Any]:
    """Serialize every structured rule, including the generic `any` anchor."""
    if rule.effect not in TARGET_EFFECT_LABELS:
        raise ValueError("V4.2.1 cannot serialize a non-target effect")
    if rule.anchor_condition == "any":
        anchor = "any"
    else:
        try:
            anchor = v42._EXTERNAL_ANCHOR[rule.anchor_condition]
        except KeyError as exc:
            raise ValueError(
                f"unsupported V4.2.1 internal rule anchor: "
                f"{rule.anchor_condition}"
            ) from exc
    payload = rule.to_dict()
    payload["anchor_condition"] = anchor
    return payload


def restore_public_rule(payload: Mapping[str, Any]) -> v4.MechanicRule:
    """Round-trip a public V4.2.1 rule into the unchanged internal engine."""
    anchor = str(payload["anchor_condition"])
    if anchor not in RULE_ANCHOR_CONDITIONS:
        raise ValueError("unsupported V4.2.1 public rule anchor")
    effect = str(payload["effect"])
    if effect not in TARGET_EFFECT_LABELS:
        raise ValueError("unsupported V4.2.1 public rule effect")
    internal_anchor = (
        "any" if anchor == "any" else v42._INTERNAL_ANCHOR[anchor]
    )
    return v4.MechanicRule(
        rule_id=str(payload["rule_id"]),
        action_scope_kind=str(payload["action_scope_kind"]),
        action_scope_value=str(payload["action_scope_value"]),
        anchor_condition=internal_anchor,
        effect=effect,
        support=int(payload.get("support", 0)),
        source=str(payload.get("source", "structured")),
    )


def _rule_round_trip(rule: v4.MechanicRule) -> bool:
    restored = restore_public_rule(public_rule(rule))
    return restored.to_dict() == rule.to_dict()


def load_frozen_manifest(
    path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    payload = v41._read_json(Path(path))
    expected = str(payload.get("manifest_checksum", ""))
    check = dict(payload)
    check.pop("manifest_checksum", None)
    actual = v41._checksum(check)
    if expected != actual:
        raise ValueError(
            f"V4.2.1 frozen-manifest checksum mismatch: {actual} != {expected}"
        )
    if payload.get("format_version") != "sage12-mechanic-induction-v4.2.1":
        raise ValueError("unsupported SAGE12 V4.2.1 manifest")
    if tuple(payload["effects"]["authoritative"]) != TARGET_EFFECT_LABELS:
        raise ValueError("V4.2.1 manifest changes authoritative effects")
    if tuple(payload["window"]["state_anchor_vocabulary"]) != (
        STATE_ANCHOR_CONDITIONS
    ):
        raise ValueError("V4.2.1 manifest changes state anchors")
    if tuple(payload["window"]["rule_anchor_vocabulary"]) != (
        RULE_ANCHOR_CONDITIONS
    ):
        raise ValueError("V4.2.1 manifest changes rule anchors")
    return payload


def _source_windows(
    frozen: Mapping[str, Any],
) -> list[v42.TargetMechanicWindowRecord]:
    traces = v41._load_traces(V3_OUTPUT_DIR / "shards", SOURCE_TRAIN)
    windows = v42.build_target_windows(
        traces,
        context_length=int(frozen["window"]["context_length"]),
    )
    for window in windows:
        v42.validate_model_view(window)
    return windows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_source_rehearsal(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    """Exercise every public rule and the complete prediction writer."""
    destination = Path(output_dir)
    frozen = load_frozen_manifest(frozen_manifest_path)
    windows = _source_windows(frozen)
    priors = v42._fit_priors(windows)
    matrices, evidence_rows = v42._raw_matrices(windows, priors)

    unique_queries = {
        (
            window.query.action_name,
            window.query.action_family,
            window.query.anchor_condition,
        ): window.query
        for window in windows
    }
    rules: list[v4.MechanicRule] = []
    for query in unique_queries.values():
        for effect in TARGET_EFFECT_LABELS:
            rules.extend(v4._rules_for_query(query.as_internal(), effect))
    round_trips = sum(_rule_round_trip(rule) for rule in rules)
    exact_any = sum(
        rule.action_scope_kind == "exact" and rule.anchor_condition == "any"
        for rule in rules
    )
    family_any = sum(
        rule.action_scope_kind == "family" and rule.anchor_condition == "any"
        for rule in rules
    )

    predictions = []
    evidence_with_any = 0
    for index, (window, evidence) in enumerate(zip(windows, evidence_rows)):
        serialized = []
        for item in evidence:
            rule = public_rule(item.rule)
            evidence_with_any += int(rule["anchor_condition"] == "any")
            serialized.append({**item.to_dict(), "rule": rule})
        predictions.append(
            {
                "window_digest": window.window_digest,
                "query": window.query.to_dict(),
                "raw_structured_probabilities": {
                    label: float(matrices["structured"][index, label_index])
                    for label_index, label in enumerate(TARGET_EFFECT_LABELS)
                },
                "evidence": serialized,
            }
        )
    predictions_path = destination / "source_rehearsal_predictions.jsonl"
    v41._write_jsonl_dicts(predictions_path, predictions)
    checks = {
        "all_rule_round_trips": round_trips == len(rules),
        "exact_any_covered": exact_any > 0,
        "family_any_covered": family_any > 0,
        "prediction_rows_complete": len(predictions) == len(windows),
        "prediction_any_serialized": evidence_with_any > 0,
        "model_view_firewall": True,
        "actor_effect_excluded": True,
    }
    payload: dict[str, Any] = {
        "format_version": REHEARSAL_FORMAT_VERSION,
        "status": "PASS_SOURCE_REHEARSAL" if all(checks.values()) else "FAIL",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "source_windows": len(windows),
        "unique_queries": len(unique_queries),
        "rules_enumerated": len(rules),
        "rule_round_trips": round_trips,
        "round_trip_rate": round_trips / max(1, len(rules)),
        "exact_any_rules": exact_any,
        "family_any_rules": family_any,
        "prediction_rows": len(predictions),
        "prediction_evidence_any_rules": evidence_with_any,
        "predictions_sha256": _file_sha256(predictions_path),
        "checks": checks,
        "source_validation_opened": False,
        "v5_protocol_authorized": False,
        "world_model_fit_authorized": False,
    }
    payload["rehearsal_checksum"] = v41._checksum(payload)
    v41._write_json(destination / "source_rehearsal.json", payload)
    return payload


def _load_passing_rehearsal(
    destination: Path,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    payload = v41._read_json(destination / "source_rehearsal.json")
    if payload.get("status") != "PASS_SOURCE_REHEARSAL":
        raise RuntimeError("V4.2.1 source rehearsal did not pass")
    if not all(dict(payload.get("checks", {})).values()):
        raise RuntimeError("V4.2.1 source rehearsal checks are incomplete")
    if payload.get("frozen_manifest_checksum") != frozen["manifest_checksum"]:
        raise RuntimeError("V4.2.1 rehearsal/manifest mismatch")
    expected = str(payload.get("rehearsal_checksum", ""))
    check = dict(payload)
    check.pop("rehearsal_checksum", None)
    if expected != v41._checksum(check):
        raise RuntimeError("V4.2.1 rehearsal checksum mismatch")
    return payload


def run_source_train_preflight(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    destination = Path(output_dir)
    frozen = load_frozen_manifest(frozen_manifest_path)
    rehearsal = _load_passing_rehearsal(destination, frozen)
    traces = v41._load_traces(V3_OUTPUT_DIR / "shards", SOURCE_TRAIN)
    windows = v42.build_target_windows(
        traces,
        context_length=int(frozen["window"]["context_length"]),
    )
    for window in windows:
        v42.validate_model_view(window)
    priors = v42._fit_priors(windows)
    calibration = v42.fit_source_calibration(windows)
    v41._write_jsonl_dicts(
        destination / "source_train_windows.jsonl",
        [window.to_dict() for window in windows],
    )
    priors_payload: dict[str, Any] = {
        "format_version": "sage12-target-mechanic-priors-v4.2.1",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "prior_strength": float(frozen["model"]["prior_strength"]),
        "internal_anchor_adapter": dict(v42._INTERNAL_ANCHOR),
        "counts": priors,
    }
    priors_payload["priors_checksum"] = v41._checksum(priors_payload)
    v41._write_json(destination / "source_priors.json", priors_payload)
    v41._write_json(destination / "calibration.json", calibration.to_dict())

    quality = v42._window_quality(windows)
    identity = v42._identity_probe(windows)
    token_budget = v42.measure_qwen_token_budget(windows, frozen)
    raw = calibration.source_oof_metrics["raw"]
    calibrated = calibration.source_oof_metrics["calibrated"]
    stronger = min(
        BASELINE_MODES,
        key=lambda mode: calibrated[mode]["macro_brier"],
    )
    source_skill = v4._brier_skill(
        calibrated["structured"],
        calibrated[stronger],
    )
    source_f1_gain = (
        calibrated["structured"]["macro_f1"]
        - calibrated[stronger]["macro_f1"]
    )
    source_context_gain = v4._brier_skill(
        calibrated["structured"],
        calibrated["context_ablation"],
    )
    gate_cfg = frozen["gates"]
    gates = {
        "source_pipeline_rehearsal": all(rehearsal["checks"].values()),
        "public_rule_round_trip": rehearsal["round_trip_rate"] == 1.0,
        "generic_any_rule_coverage": (
            rehearsal["exact_any_rules"] > 0
            and rehearsal["family_any_rules"] > 0
            and rehearsal["prediction_evidence_any_rules"] > 0
        ),
        "minimum_source_train_windows": len(windows)
        >= int(gate_cfg["minimum_source_train_windows"]),
        "source_train_label_capacity": v42._label_capacity(
            quality["per_label"],
            int(gate_cfg["minimum_source_train_positives_per_label"]),
            int(gate_cfg["minimum_source_train_negatives_per_label"]),
        ),
        "static_identity_leakage": identity["gain"]
        <= float(gate_cfg["maximum_static_identity_gain_over_action"]),
        "source_oof_calibration": calibrated["structured"]["macro_ece"]
        <= float(gate_cfg["maximum_source_oof_macro_ece"]),
        "source_oof_brier_non_degradation": (
            calibrated["structured"]["macro_brier"]
            - raw["structured"]["macro_brier"]
        )
        <= float(gate_cfg["maximum_source_oof_brier_degradation"]),
        "source_oof_brier_skill": source_skill
        >= float(gate_cfg["minimum_source_macro_brier_skill"]),
        "source_oof_macro_f1_gain": source_f1_gain
        >= float(gate_cfg["minimum_source_macro_f1_gain"]),
        "source_context_brier_skill_gain": source_context_gain
        >= float(gate_cfg["minimum_source_context_brier_skill_gain"]),
        "qwen_prompt_budget": token_budget["maximum_tokens"]
        <= int(frozen["qwen"]["preflight_maximum_input_tokens"]),
        "model_view_firewall": True,
        "actor_effect_excluded": True,
    }
    payload: dict[str, Any] = {
        "format_version": PREFLIGHT_FORMAT_VERSION,
        "status": (
            "PASS_SOURCE_TRAIN_PREFLIGHT"
            if all(gates.values())
            else "FAIL_SOURCE_TRAIN_PREFLIGHT"
        ),
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "rehearsal_checksum": rehearsal["rehearsal_checksum"],
        "rows": len(traces),
        "windows": len(windows),
        "quality": quality,
        "identity_probe": identity,
        "source_oof": calibration.source_oof_metrics,
        "source_stronger_baseline": stronger,
        "source_macro_brier_skill": source_skill,
        "source_macro_f1_gain": source_f1_gain,
        "source_context_brier_skill_gain": source_context_gain,
        "calibration_checksum": calibration.calibration_checksum,
        "priors_checksum": priors_payload["priors_checksum"],
        "qwen_token_budget": token_budget,
        "gates": gates,
        "source_validation_opened": False,
        "v5_protocol_authorized": False,
        "world_model_fit_authorized": False,
    }
    payload["preflight_checksum"] = v41._checksum(payload)
    v41._write_json(destination / "source_train_preflight.json", payload)
    return payload


def _validate_collection(
    destination: Path,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    payload = v41._read_json(destination / "collection_manifest.json")
    if payload.get("format_version") != (
        "sage12-target-mechanic-collection-v4.2.1"
    ):
        raise ValueError("unsupported V4.2.1 collection manifest")
    if payload.get("status") != "COMPLETE":
        raise ValueError("V4.2.1 collection is incomplete")
    if payload.get("frozen_manifest_checksum") != frozen["manifest_checksum"]:
        raise ValueError("V4.2.1 collection/manifest mismatch")
    if int(payload.get("rows", 0)) != int(
        frozen["collection"]["prospective_rows"]
    ):
        raise ValueError("V4.2.1 collection row count mismatch")
    expected = str(payload.get("report_checksum", ""))
    check = dict(payload)
    check.pop("report_checksum", None)
    if expected != v41._checksum(check):
        raise ValueError("V4.2.1 collection checksum mismatch")
    return payload


def _write_predictions(
    destination: Path,
    windows: Sequence[v42.TargetMechanicWindowRecord],
    raw_matrices: Mapping[str, np.ndarray],
    calibrated_matrices: Mapping[str, np.ndarray],
    evidence_rows: Sequence[Sequence[v4.MechanicEvidence]],
) -> dict[str, Any]:
    rows = []
    any_rules = 0
    for index, (window, evidence) in enumerate(zip(windows, evidence_rows)):
        serialized_evidence = []
        for item in evidence:
            rule = public_rule(item.rule)
            any_rules += int(rule["anchor_condition"] == "any")
            serialized_evidence.append({**item.to_dict(), "rule": rule})
        rows.append(
            {
                "window_digest": window.window_digest,
                "game_id": window.game_id,
                "run_key": window.run_key,
                "query": window.query.to_dict(),
                "labels": dict(window.labels),
                "applicable": dict(window.applicable),
                "raw_structured_probabilities": {
                    label: float(raw_matrices["structured"][index, label_index])
                    for label_index, label in enumerate(TARGET_EFFECT_LABELS)
                },
                "calibrated_structured_probabilities": {
                    label: float(
                        calibrated_matrices["structured"][index, label_index]
                    )
                    for label_index, label in enumerate(TARGET_EFFECT_LABELS)
                },
                "evidence": serialized_evidence,
            }
        )
    path = destination / "predictions.jsonl"
    v41._write_jsonl_dicts(path, rows)
    return {
        "rows": len(rows),
        "any_rules": any_rules,
        "sha256": _file_sha256(path),
    }


def _build_runtime_failure(
    destination: Path,
    frozen: Mapping[str, Any],
    preflight: Mapping[str, Any] | None,
    state: Mapping[str, str],
    exc: Exception,
) -> dict[str, Any]:
    artifacts = {}
    for name in (
        "validation_windows.jsonl",
        "predictions.jsonl",
        "structured_intermediate.json",
        "qwen_outputs.jsonl",
        "qwen_outcome_shuffle_outputs.jsonl",
    ):
        path = destination / name
        if path.exists():
            artifacts[name] = {
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
    payload: dict[str, Any] = {
        "format_version": FAILURE_FORMAT_VERSION,
        "status": "FAIL_RUNTIME_CLOSED",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "preflight_checksum": (
            preflight.get("preflight_checksum") if preflight else None
        ),
        "stage": state["stage"],
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "artifacts": artifacts,
        "prospective_outcomes_opened": True,
        "structured_verdict_available": (
            "structured_intermediate.json" in artifacts
        ),
        "qwen_verdict_available": False,
        "v5_protocol_authorized": False,
        "world_model_fit_authorized": False,
        "ebm_fit_authorized": False,
    }
    payload["result_checksum"] = v41._checksum(payload)
    v41._write_json(destination / "runtime_failure.json", payload)
    return payload


def _run_evaluation(
    *,
    output_dir: str | Path,
    frozen_manifest_path: str | Path,
    run_qwen: bool,
    state: dict[str, str],
) -> dict[str, Any]:
    destination = Path(output_dir)
    state["stage"] = "load_frozen_inputs"
    frozen = load_frozen_manifest(frozen_manifest_path)
    _load_passing_rehearsal(destination, frozen)
    preflight = v41._read_json(destination / "source_train_preflight.json")
    if (
        preflight.get("status") != "PASS_SOURCE_TRAIN_PREFLIGHT"
        or not all(dict(preflight.get("gates", {})).values())
    ):
        raise RuntimeError("V4.2.1 preflight did not authorize evaluation")
    if preflight.get("frozen_manifest_checksum") != frozen["manifest_checksum"]:
        raise RuntimeError("V4.2.1 preflight/manifest mismatch")
    _validate_collection(destination, frozen)
    priors = dict(
        v41._read_json(destination / "source_priors.json")["counts"]
    )
    bundle = v42.TargetCalibrationBundle.from_dict(
        v41._read_json(destination / "calibration.json")
    )

    state["stage"] = "build_validation_windows"
    traces = v41._load_traces(destination / "shards", SOURCE_VALIDATION)
    windows = v42.build_target_windows(
        traces,
        context_length=int(frozen["window"]["context_length"]),
    )
    for window in windows:
        v42.validate_model_view(window)
    v41._write_jsonl_dicts(
        destination / "validation_windows.jsonl",
        [window.to_dict() for window in windows],
    )

    state["stage"] = "structured_evaluation"
    targets, masks = v42._targets_masks(windows)
    raw_matrices, evidence_rows = v42._raw_matrices(windows, priors)
    calibrated_matrices = {
        mode: v42.apply_calibration(matrix, bundle, mode)
        for mode, matrix in raw_matrices.items()
    }
    raw_metrics = {
        mode: v42.multilabel_metrics(targets, masks, matrix)
        for mode, matrix in raw_matrices.items()
    }
    calibrated_metrics = {
        mode: v42.multilabel_metrics(
            targets,
            masks,
            matrix,
            thresholds=bundle.thresholds[mode],
        )
        for mode, matrix in calibrated_matrices.items()
    }
    stronger = min(
        BASELINE_MODES,
        key=lambda mode: calibrated_metrics[mode]["macro_brier"],
    )
    raw_skill = v4._brier_skill(
        raw_metrics["structured"],
        raw_metrics[stronger],
    )
    calibrated_skill = v4._brier_skill(
        calibrated_metrics["structured"],
        calibrated_metrics[stronger],
    )
    f1_gain = (
        calibrated_metrics["structured"]["macro_f1"]
        - calibrated_metrics[stronger]["macro_f1"]
    )

    outcome_windows = v42._shuffle_context(windows, binding=False)
    outcome_raw, _ = v42._raw_matrices(outcome_windows, priors)
    outcome_matrix = v42.apply_calibration(
        outcome_raw["structured"],
        bundle,
        "structured",
    )
    outcome_metrics = v42.multilabel_metrics(
        targets,
        masks,
        outcome_matrix,
        thresholds=bundle.thresholds["structured"],
    )
    outcome_drop = calibrated_skill - v4._brier_skill(
        outcome_metrics,
        calibrated_metrics[stronger],
    )
    binding_windows = v42._shuffle_context(windows, binding=True)
    binding_raw, _ = v42._raw_matrices(binding_windows, priors)
    binding_matrix = v42.apply_calibration(
        binding_raw["structured"],
        bundle,
        "structured",
    )
    binding_metrics = v42.multilabel_metrics(
        targets,
        masks,
        binding_matrix,
        thresholds=bundle.thresholds["structured"],
    )
    binding_skill = v4._brier_skill(
        binding_metrics,
        calibrated_metrics[stronger],
    )
    binding_drop = calibrated_skill - binding_skill
    context_gain = v4._brier_skill(
        calibrated_metrics["structured"],
        calibrated_metrics["context_ablation"],
    )
    bootstrap = v42._bootstrap_skill(
        windows,
        targets,
        masks,
        calibrated_matrices["structured"],
        calibrated_matrices[stronger],
        model_thresholds=bundle.thresholds["structured"],
        baseline_thresholds=bundle.thresholds[stronger],
        samples=int(frozen["evaluation"]["bootstrap_samples"]),
        seed=int(frozen["evaluation"]["random_seed"]),
    )

    per_game = {}
    for game in SOURCE_VALIDATION:
        selected = np.asarray([window.game_id == game for window in windows])
        model_metric = v42.multilabel_metrics(
            targets[selected],
            masks[selected],
            calibrated_matrices["structured"][selected],
            thresholds=bundle.thresholds["structured"],
        )
        candidates = {
            mode: v42.multilabel_metrics(
                targets[selected],
                masks[selected],
                calibrated_matrices[mode][selected],
                thresholds=bundle.thresholds[mode],
            )
            for mode in BASELINE_MODES
        }
        game_baseline = min(
            BASELINE_MODES,
            key=lambda mode: candidates[mode]["macro_brier"],
        )
        per_game[game] = {
            "windows": int(np.sum(selected)),
            "structured": model_metric,
            "stronger_baseline": game_baseline,
            "baseline": candidates[game_baseline],
            "brier_skill": v4._brier_skill(
                model_metric,
                candidates[game_baseline],
            ),
        }
    quality = v42._window_quality(windows)
    identity = v42._identity_probe(windows)
    output_contract = v42._output_contract(windows, evidence_rows)
    effect_authority = v42._effect_authority(calibrated_metrics, quality)
    gate_cfg = frozen["gates"]
    gates = {
        "minimum_prospective_windows": len(windows)
        >= int(gate_cfg["minimum_prospective_windows"]),
        "prospective_label_capacity": v42._label_capacity(
            quality["per_label"],
            int(gate_cfg["minimum_validation_positives_per_label"]),
            int(gate_cfg["minimum_validation_negatives_per_label"]),
        ),
        "strict_json_validity": output_contract["strict_json_validity"] == 1.0,
        "support_zero_rate": output_contract["support_zero_rate"] == 1.0,
        "grounded_hypothesis_rate": output_contract[
            "grounded_hypothesis_rate"
        ]
        == 1.0,
        "minimum_raw_brier_skill": raw_skill
        >= float(gate_cfg["minimum_macro_brier_skill"]),
        "minimum_calibrated_brier_skill": calibrated_skill
        >= float(gate_cfg["minimum_macro_brier_skill"]),
        "bootstrap_lower_bound_positive": bootstrap["lower_95"] > 0.0,
        "minimum_macro_f1_gain": f1_gain
        >= float(gate_cfg["minimum_macro_f1_gain"]),
        "minimum_outcome_shuffle_drop": outcome_drop
        >= float(gate_cfg["minimum_outcome_shuffle_skill_drop"]),
        "minimum_binding_shuffle_drop": binding_drop
        >= float(gate_cfg["minimum_binding_shuffle_skill_drop"]),
        "minimum_context_gain": context_gain
        >= float(gate_cfg["minimum_context_brier_skill_gain"]),
        "every_game_nonnegative": all(
            item["brier_skill"] >= 0.0 for item in per_game.values()
        ),
        "maximum_macro_ece": calibrated_metrics["structured"]["macro_ece"]
        <= float(gate_cfg["maximum_macro_ece"]),
        "maximum_prospective_identity_gain": identity["gain"]
        <= float(gate_cfg["maximum_static_identity_gain_over_action"]),
        "effect_authority": all(
            effect_authority[label]["eligible_for_v5"]
            for label in TARGET_EFFECT_LABELS
        ),
        "source_preflight_passed": True,
        "source_rehearsal_passed": True,
        "actor_effect_excluded": True,
    }
    structured_passed = all(gates.values())

    state["stage"] = "prediction_serialization"
    prediction_artifact = _write_predictions(
        destination,
        windows,
        raw_matrices,
        calibrated_matrices,
        evidence_rows,
    )
    if prediction_artifact["rows"] != len(windows):
        raise RuntimeError("V4.2.1 prediction writer lost rows")
    if prediction_artifact["any_rules"] <= 0:
        raise RuntimeError("V4.2.1 prediction writer did not exercise any")

    structured_payload: dict[str, Any] = {
        "format_version": "sage12-target-mechanic-structured-v4.2.1",
        "status": "PASS" if structured_passed else "FAIL_CLOSED",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "preflight_checksum": preflight["preflight_checksum"],
        "rows": {
            "prospective_transitions": len(traces),
            "prospective_windows": len(windows),
        },
        "prediction_artifact": prediction_artifact,
        "quality": quality,
        "identity_probe": identity,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "stronger_baseline": stronger,
        "raw_macro_brier_skill": raw_skill,
        "calibrated_macro_brier_skill": calibrated_skill,
        "calibrated_macro_f1_gain": f1_gain,
        "bootstrap_skill": bootstrap,
        "outcome_shuffle": {
            "metrics": outcome_metrics,
            "skill_drop": outcome_drop,
        },
        "binding_shuffle": {
            "metrics": binding_metrics,
            "skill": binding_skill,
            "skill_drop": binding_drop,
        },
        "context_brier_skill_gain": context_gain,
        "per_game": per_game,
        "effect_authority": effect_authority,
        "output_contract": output_contract,
        "gates": gates,
        "v5_protocol_authorized": structured_passed,
        "world_model_fit_authorized": False,
        "ebm_fit_authorized": False,
    }
    structured_payload["structured_checksum"] = v41._checksum(
        structured_payload
    )
    v41._write_json(
        destination / "structured_intermediate.json",
        structured_payload,
    )

    state["stage"] = "qwen_evaluation"
    if run_qwen:
        try:
            qwen = v42._evaluate_qwen(
                windows,
                priors,
                bundle,
                frozen,
                targets,
                masks,
                calibrated_matrices[stronger],
                stronger,
                output_dir=destination,
            )
        except Exception as exc:  # noqa: BLE001 - separate authority branch
            qwen = {
                "status": "FAIL_RUNTIME",
                "authority_separate": True,
                "error": f"{type(exc).__name__}: {exc}",
                "device": str(frozen["qwen"]["device"]),
            }
    else:
        qwen = {"status": "SKIPPED", "authority_separate": True}

    state["stage"] = "final_result_serialization"
    payload: dict[str, Any] = {
        "format_version": RESULT_FORMAT_VERSION,
        "status": "PASS" if structured_passed else "FAIL_CLOSED",
        "all_structured_gates_passed": structured_passed,
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "preflight_checksum": preflight["preflight_checksum"],
        "calibration_checksum": bundle.calibration_checksum,
        "structured_checksum": structured_payload["structured_checksum"],
        "rows": structured_payload["rows"],
        "prediction_artifact": prediction_artifact,
        "quality": quality,
        "identity_probe": identity,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "stronger_baseline": stronger,
        "raw_macro_brier_skill": raw_skill,
        "calibrated_macro_brier_skill": calibrated_skill,
        "calibrated_macro_f1_gain": f1_gain,
        "bootstrap_skill": bootstrap,
        "outcome_shuffle": structured_payload["outcome_shuffle"],
        "binding_shuffle": structured_payload["binding_shuffle"],
        "context_brier_skill_gain": context_gain,
        "per_game": per_game,
        "effect_authority": effect_authority,
        "output_contract": output_contract,
        "qwen": qwen,
        "gates": gates,
        "firewall": {
            "source_only_calibration": True,
            "actor_effect_modelled": False,
            "v4_2_shards_reused": False,
            "holdout_opened": False,
            "historical_opened": False,
            "ar25_opened": False,
            "controller_executed": False,
        },
        "v5_protocol_authorized": structured_passed,
        "qwen_v5_protocol_authorized": bool(
            structured_passed and qwen.get("status") == "PASS"
        ),
        "world_model_fit_authorized": False,
        "ebm_fit_authorized": False,
    }
    payload["result_checksum"] = v41._checksum(payload)
    v41._write_json(destination / "pilot_result.json", payload)
    state["stage"] = "complete"
    return payload


def run_evaluation(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    run_qwen: bool = True,
) -> dict[str, Any]:
    state = {"stage": "initialization"}
    destination = Path(output_dir)
    frozen: dict[str, Any] | None = None
    preflight: dict[str, Any] | None = None
    try:
        frozen = load_frozen_manifest(frozen_manifest_path)
        preflight_path = destination / "source_train_preflight.json"
        if preflight_path.exists():
            preflight = v41._read_json(preflight_path)
        return _run_evaluation(
            output_dir=destination,
            frozen_manifest_path=frozen_manifest_path,
            run_qwen=run_qwen,
            state=state,
        )
    except Exception as exc:
        if frozen is None:
            raise
        return _build_runtime_failure(
            destination,
            frozen,
            preflight,
            state,
            exc,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("rehearsal", "preflight", "evaluate"),
        default="rehearsal",
    )
    parser.add_argument(
        "--frozen-manifest",
        default=str(DEFAULT_FROZEN_MANIFEST_PATH),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-qwen", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "rehearsal":
        result = run_source_rehearsal(
            output_dir=args.output_dir,
            frozen_manifest_path=args.frozen_manifest,
        )
    elif args.command == "preflight":
        result = run_source_train_preflight(
            output_dir=args.output_dir,
            frozen_manifest_path=args.frozen_manifest,
        )
    else:
        result = run_evaluation(
            output_dir=args.output_dir,
            frozen_manifest_path=args.frozen_manifest,
            run_qwen=not args.skip_qwen,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FROZEN_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "RULE_ANCHOR_CONDITIONS",
    "STATE_ANCHOR_CONDITIONS",
    "TARGET_EFFECT_LABELS",
    "compact_qwen_prompt",
    "compact_qwen_schema",
    "load_frozen_manifest",
    "public_rule",
    "restore_public_rule",
    "run_evaluation",
    "run_source_rehearsal",
    "run_source_train_preflight",
]
