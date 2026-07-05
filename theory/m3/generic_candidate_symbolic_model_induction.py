"""M3.G0.7 candidate-only symbolic model induction."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .generic_mechanic_contextual_evidence_consolidation import (
    DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_EVIDENCE_CONSOLIDATION_OUTPUT_PATH,
    CONTEXT_STABLE_CANDIDATE_ONLY,
    TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY,
    TEMPORAL_REGULARITY_FAILED_CANDIDATE_ONLY,
)
from .generic_mechanic_contextual_followup_planner import (
    DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_FOLLOWUP_REQUESTS_OUTPUT_PATH,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_GENERIC_CANDIDATE_SYMBOLIC_MODEL_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "generic_candidate_symbolic_mechanism_model.json"
)
SCHEMA_VERSION = "m3.generic_candidate_symbolic_mechanism_model.v1"


def run_generic_candidate_symbolic_model_induction(
    *,
    contextual_consolidation_path: str | Path = (
        DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_EVIDENCE_CONSOLIDATION_OUTPUT_PATH
    ),
    contextual_requests_path: str | Path | None = None,
) -> Dict[str, Any]:
    consolidation_payload = _load_json(contextual_consolidation_path)
    validate_contextual_consolidation_source(consolidation_payload)
    request_path = Path(
        contextual_requests_path
        or resolve_contextual_requests_path(consolidation_payload)
    )
    request_payload = _load_json(request_path)
    requests_by_id = {
        str(row.get("request_id", "")): dict(row)
        for row in request_payload.get("generic_contextual_followup_requests", []) or []
    }
    hypothesis_records = [
        dict(row)
        for row in consolidation_payload.get("hypothesis_consolidations", []) or []
    ]
    stable_records = [
        row
        for row in hypothesis_records
        if bool(row.get("ready_for_symbolic_model_candidate_only"))
    ]
    caveat_records = [
        row
        for row in hypothesis_records
        if str(row.get("candidate_status", ""))
        not in {CONTEXT_STABLE_CANDIDATE_ONLY, TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY}
    ]
    model = {
        "candidate_symbolic_model_id": "m3g0_7::bp35::generic_symbolic_mechanism_model",
        "model_status": "CANDIDATE_ONLY",
        "model_kind": "generic_symbolic_mechanism_model",
        "source_contextual_consolidation_path": str(contextual_consolidation_path),
        "source_contextual_requests_path": str(request_path),
        "actor_candidates": actor_candidates(stable_records, requests_by_id),
        "action_models": action_models(stable_records, requests_by_id),
        "relation_model": relation_model(stable_records, requests_by_id),
        "dynamic_invariants": dynamic_invariants(stable_records, requests_by_id),
        "caveats": caveats(caveat_records, requests_by_id),
        "semantic_interpretation": "unknown",
        "model_counted_as_confirmation": False,
        "symbolic_model_induction_performed": True,
        "symbolic_model_induction_counted_as_verdict": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    return {
        "config": {
            "contextual_consolidation_path": str(contextual_consolidation_path),
            "contextual_requests_path": str(request_path),
            "schema_version": SCHEMA_VERSION,
            "inputs_read": ["M3.G0.6", "M3.G0.4"],
            "artifacts_not_modified": ["M1", "M2", "A32", "A33"],
            "execution_performed": False,
            "induction_policy": {
                "stable_contextual_candidates_become_model_parts": True,
                "model_parts_are_not_scientific_truth": True,
                "dynamic_invariant_semantics_remain_unknown": True,
                "caveats_are_preserved": True,
            },
        },
        "summary": summarize_symbolic_model(
            consolidation_payload=consolidation_payload,
            model=model,
        ),
        "candidate_symbolic_model": model,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "model_counted_as_confirmation": False,
        "symbolic_model_induction_counted_as_verdict": False,
        "semantic_interpretation_counted_as_confirmation": False,
        "a32_remains_only_verdict_location": True,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def actor_candidates(
    records: Sequence[Mapping[str, Any]],
    requests_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    by_entity: dict[str, Dict[str, Any]] = {}
    for record in records:
        if str(record.get("source_mechanic_family", "")) != "entity_role":
            continue
        source = request_for_record(record, requests_by_id)
        if str(source.get("candidate_role", "")) != "controllable_actor":
            continue
        entity_id = str(source.get("target_entity", ""))
        if not entity_id:
            continue
        row = by_entity.setdefault(
            entity_id,
            {
                "entity_id": entity_id,
                "role": "controllable_actor_candidate",
                "basis": CONTEXT_STABLE_CANDIDATE_ONLY,
                "source_hypothesis_ids": [],
                "source_request_ids": [],
                "observed_contextual_tests": 0,
                "semantic_interpretation": "unknown",
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
            },
        )
        row["source_hypothesis_ids"].append(str(record.get("source_hypothesis_id", "")))
        row["source_request_ids"].append(str(record.get("request_id", "")))
        row["observed_contextual_tests"] += 1
    return tuple(_dedupe_list_fields(row) for row in by_entity.values())


def action_models(
    records: Sequence[Mapping[str, Any]],
    requests_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    models: dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "candidate_effects": [],
            "relation_effects": [],
            "source_hypothesis_ids": [],
            "source_request_ids": [],
            "status": CONTEXT_STABLE_CANDIDATE_ONLY,
            "semantic_interpretation": "unknown",
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        }
    )
    for record in records:
        source = request_for_record(record, requests_by_id)
        family = str(record.get("source_mechanic_family", ""))
        action = _optional_text(source.get("target_action"))
        if not action:
            continue
        model = models[action]
        model["source_hypothesis_ids"].append(str(record.get("source_hypothesis_id", "")))
        model["source_request_ids"].append(str(record.get("request_id", "")))
        if family == "action_effect" and source.get("predicted_effect_family"):
            model["candidate_effects"].append(str(source.get("predicted_effect_family")))
        if family == "relation_change" and source.get("relation_delta_type"):
            model["relation_effects"].append(str(source.get("relation_delta_type")))
    return {
        action: _dedupe_list_fields(dict(row))
        for action, row in sorted(models.items())
    }


def relation_model(
    records: Sequence[Mapping[str, Any]],
    requests_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    effects: list[Dict[str, Any]] = []
    for record in records:
        if str(record.get("source_mechanic_family", "")) != "relation_change":
            continue
        source = request_for_record(record, requests_by_id)
        effect = {
            "source_entity": str(source.get("source_entity", "")),
            "target_entity": str(source.get("relation_target_entity", "")),
            "action": str(source.get("target_action", "")),
            "relation_delta_type": str(source.get("relation_delta_type", "")),
            "status": CONTEXT_STABLE_CANDIDATE_ONLY,
            "source_hypothesis_id": str(record.get("source_hypothesis_id", "")),
            "source_request_id": str(record.get("request_id", "")),
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        }
        if effect["source_entity"] and effect["relation_delta_type"]:
            effects.append(effect)
    return {
        "actor_relation_effects": effects,
        "semantic_interpretation": "unknown",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }


def dynamic_invariants(
    records: Sequence[Mapping[str, Any]],
    requests_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    invariants: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if str(record.get("candidate_status", "")) != TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY:
            continue
        source = request_for_record(record, requests_by_id)
        entity_id = str(source.get("target_entity", ""))
        if not entity_id:
            continue
        invariants[entity_id] = {
            "entity_id": entity_id,
            "family": str(source.get("invariant_family", "")),
            "invariant_id": str(source.get("invariant_id", "")),
            "status": TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY,
            "semantic_interpretation": "unknown",
            "source_hypothesis_id": str(record.get("source_hypothesis_id", "")),
            "source_request_id": str(record.get("request_id", "")),
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        }
    return dict(sorted(invariants.items()))


def caveats(
    records: Sequence[Mapping[str, Any]],
    requests_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    rows: list[Dict[str, Any]] = []
    for record in records:
        source = request_for_record(record, requests_by_id)
        rows.append(
            {
                "entity_id": str(source.get("target_entity", "")),
                "family": str(source.get("invariant_family", "")),
                "candidate_status": str(record.get("candidate_status", "")),
                "source_hypothesis_id": str(record.get("source_hypothesis_id", "")),
                "source_request_id": str(record.get("request_id", "")),
                "followup_required": bool(record.get("followup_required_for_contradiction")),
                "semantic_interpretation": "unknown",
                "contradiction_counted_as_refutation": False,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
            }
        )
    return tuple(rows)


def summarize_symbolic_model(
    *,
    consolidation_payload: Mapping[str, Any],
    model: Mapping[str, Any],
) -> Dict[str, Any]:
    source_summary = dict(consolidation_payload.get("summary", {}) or {})
    action_models_value = dict(model.get("action_models", {}) or {})
    relation_value = dict(model.get("relation_model", {}) or {})
    dynamic_value = dict(model.get("dynamic_invariants", {}) or {})
    caveat_value = list(model.get("caveats", []) or [])
    return {
        "source_hypothesis_consolidations": int(
            source_summary.get("hypothesis_consolidations", 0) or 0
        ),
        "source_context_diversity_assessment": str(
            source_summary.get("context_diversity_assessment", "")
        ),
        "source_independent_contexts": int(source_summary.get("independent_contexts", 0) or 0),
        "actor_candidates": len(model.get("actor_candidates", []) or []),
        "action_models": len(action_models_value),
        "relation_effects": len(relation_value.get("actor_relation_effects", []) or []),
        "dynamic_invariants": len(dynamic_value),
        "caveats": len(caveat_value),
        "ready_for_policy_probe_candidate_only": bool(
            model.get("actor_candidates")
            and action_models_value
            and relation_value.get("actor_relation_effects")
        ),
        "model_status": "CANDIDATE_ONLY",
        "model_counted_as_confirmation": False,
        "symbolic_model_induction_performed": True,
        "symbolic_model_induction_counted_as_verdict": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def validate_contextual_consolidation_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    readiness = dict(payload.get("symbolic_model_readiness", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("M3.G0.6 support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("M3.G0.6 summary support must remain 0")
    if bool(payload.get("revision_performed", False)):
        raise ValueError("M3.G0.6 must not perform revision")
    if bool(payload.get("contextual_events_counted_as_scientific_support", False)):
        raise ValueError("contextual events cannot be scientific support")
    if bool(payload.get("semantic_interpretation_counted_as_confirmation", False)):
        raise ValueError("semantic interpretation cannot be confirmation")
    if bool(payload.get("contradiction_counted_as_refutation", False)):
        raise ValueError("contradiction cannot be refutation in M3.G0.6")
    if bool(payload.get("a32_write_performed", False)) or bool(payload.get("a33_write_performed", False)):
        raise ValueError("M3.G0.6 cannot write A32/A33")
    if not bool(readiness.get("ready_for_symbolic_model_candidate_only")):
        raise ValueError("M3.G0.6 source is not ready for symbolic model candidate")


def resolve_contextual_requests_path(
    consolidation_payload: Mapping[str, Any],
) -> Path:
    contextual_results_path = str(
        (consolidation_payload.get("config", {}) or {}).get("contextual_results_path", "")
    )
    if contextual_results_path and Path(contextual_results_path).exists():
        results_payload = _load_json(contextual_results_path)
        request_path = str(
            (results_payload.get("config", {}) or {}).get("contextual_requests_path", "")
        )
        if request_path:
            return Path(request_path)
    return DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_FOLLOWUP_REQUESTS_OUTPUT_PATH


def request_for_record(
    record: Mapping[str, Any],
    requests_by_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    return requests_by_id.get(str(record.get("request_id", "")), {})


def _dedupe_list_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    clean = dict(row)
    for key, value in list(clean.items()):
        if isinstance(value, list):
            clean[key] = list(dict.fromkeys(str(item) for item in value if item))
    return clean


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text if text and text != "None" else ""


def write_generic_candidate_symbolic_model(
    payload: Mapping[str, Any],
    out_path: str | Path = DEFAULT_GENERIC_CANDIDATE_SYMBOLIC_MODEL_OUTPUT_PATH,
) -> None:
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Induce candidate-only symbolic generic mechanism model."
    )
    parser.add_argument(
        "--contextual-consolidation",
        default=str(DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_EVIDENCE_CONSOLIDATION_OUTPUT_PATH),
        help="Path to diagnostics/m3/generic_mechanic_contextual_evidence_consolidation.json.",
    )
    parser.add_argument(
        "--contextual-requests",
        default=None,
        help="Optional path to diagnostics/m3/generic_mechanic_contextual_followup_requests.json.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_GENERIC_CANDIDATE_SYMBOLIC_MODEL_OUTPUT_PATH),
        help="Output path for M3.G0.7 candidate symbolic model.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_generic_candidate_symbolic_model_induction(
        contextual_consolidation_path=args.contextual_consolidation,
        contextual_requests_path=args.contextual_requests,
    )
    write_generic_candidate_symbolic_model(payload, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "actor_candidates": payload["summary"]["actor_candidates"],
                "action_models": payload["summary"]["action_models"],
                "relation_effects": payload["summary"]["relation_effects"],
                "dynamic_invariants": payload["summary"]["dynamic_invariants"],
                "caveats": payload["summary"]["caveats"],
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
