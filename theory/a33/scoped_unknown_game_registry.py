"""A33.2 registry for scope-locked unknown-game mechanics confirmed by A32.5."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from theory.a32.unknown_game_parameterized_control_revision_decisions import (
    A32_5_MIXED_SCOPE_CONFIRMATION_AND_NON_IDENTIFIABILITY,
    A32_5_SCHEMA_VERSION,
    CONFIRM_SCOPE_LIMITED_AFTER_PARAMETERIZED_CONTROL_REVISION,
    DEFAULT_A32_UNKNOWN_GAME_PARAMETERIZED_CONTROL_REVISION_OUTPUT_PATH,
    KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_CONTROL,
)

from .confirmed_mechanics_registry import mechanic_prediction_spec


DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH = (
    Path("diagnostics") / "a33" / "scoped_unknown_game_registry.json"
)

A33_2_SCHEMA_VERSION = "a33.scoped_unknown_game_registry.v1"
A33_2_TRUTH_STATUS = "NOT_REEVALUATED_BY_A33_2"
A33_2_ENTRY_ADDED = "A33_SCOPED_UNKNOWN_GAME_REGISTRY_ENTRY_ADDED"
A33_2_NO_ELIGIBLE_ENTRY = "A33_NO_ELIGIBLE_SCOPED_UNKNOWN_GAME_ENTRY"


@dataclass(frozen=True)
class ScopedUnknownGameRegistryEntry:
    """One A32-confirmed mechanic kept inside its exact unknown-game scope."""

    key: str
    candidate_id: str
    game_id: str
    action: str
    action_args: Dict[str, Any] | None
    mechanic_family: str
    predicted_metric: str
    confirmed_support: int
    contradictions: int
    experiments_spent: int
    budgets: Tuple[int, ...]
    context_snapshot_hashes: Tuple[str, ...]
    parameterized_control_variants: Tuple[Dict[str, Any], ...]
    source_experiment_ids: Tuple[str, ...]
    scoped_claim: str
    source_artifact: str
    known_scope: str = "game_candidate_contexts_measurement"
    source_decision: str = (
        CONFIRM_SCOPE_LIMITED_AFTER_PARAMETERIZED_CONTROL_REVISION
    )
    status: str = "confirmed"
    evidence_notes: Tuple[str, ...] = field(default_factory=tuple)
    scope_game_locked: bool = True
    scope_candidate_locked: bool = True
    scope_contexts_locked: bool = True
    scope_measurement_locked: bool = True
    not_generalized_beyond_game: bool = True
    not_generalized_beyond_candidate_scope: bool = True
    not_generalized_to_other_actions: bool = True
    a32_confirmation_reused_without_reevaluation: bool = True
    a33_confirmation_performed: bool = False
    unresolved_candidates_excluded: bool = True
    non_identifiable_candidates_excluded: bool = True
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "candidate_id": self.candidate_id,
            "game_id": self.game_id,
            "action": self.action,
            "action_args": (
                dict(self.action_args) if self.action_args is not None else None
            ),
            "mechanic_family": self.mechanic_family,
            "predicted_metric": self.predicted_metric,
            "confirmed_support": int(self.confirmed_support),
            "contradictions": int(self.contradictions),
            "experiments_spent": int(self.experiments_spent),
            "budgets": list(self.budgets),
            "context_snapshot_hashes": list(self.context_snapshot_hashes),
            "parameterized_control_variants": [
                dict(row) for row in self.parameterized_control_variants
            ],
            "source_experiment_ids": list(self.source_experiment_ids),
            "scoped_claim": self.scoped_claim,
            "source_artifact": self.source_artifact,
            "known_scope": self.known_scope,
            "source_decision": self.source_decision,
            "status": self.status,
            "evidence_notes": list(self.evidence_notes),
            "scope_game_locked": self.scope_game_locked,
            "scope_candidate_locked": self.scope_candidate_locked,
            "scope_contexts_locked": self.scope_contexts_locked,
            "scope_measurement_locked": self.scope_measurement_locked,
            "not_generalized_beyond_game": self.not_generalized_beyond_game,
            "not_generalized_beyond_candidate_scope": (
                self.not_generalized_beyond_candidate_scope
            ),
            "not_generalized_to_other_actions": self.not_generalized_to_other_actions,
            "a32_confirmation_reused_without_reevaluation": (
                self.a32_confirmation_reused_without_reevaluation
            ),
            "a33_confirmation_performed": self.a33_confirmation_performed,
            "unresolved_candidates_excluded": self.unresolved_candidates_excluded,
            "non_identifiable_candidates_excluded": (
                self.non_identifiable_candidates_excluded
            ),
            "wrong_confirmations": int(self.wrong_confirmations),
        }


def run_scoped_unknown_game_registry_generation(
    *,
    decisions_path: str | Path = (
        DEFAULT_A32_UNKNOWN_GAME_PARAMETERIZED_CONTROL_REVISION_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    """Validate A32.5 and register only its explicitly eligible handoffs."""
    source = _load_json(decisions_path)
    entries = build_scoped_unknown_game_registry(
        source,
        source_artifact=str(decisions_path),
    )
    excluded = build_excluded_candidate_audit(source)
    summary = summarize_scoped_unknown_game_registry(entries, excluded)
    return {
        "config": {
            "decisions_path": str(decisions_path),
            "schema_version": A33_2_SCHEMA_VERSION,
            "inputs_read": ["A32.5"],
            "registry_scope": "A33_SCOPE_LOCKED_UNKNOWN_GAME_CONFIRMED_MEMORY",
            "artifacts_not_modified": [
                "A32.5",
                "A33.1",
                "M2",
                "M3",
                "A34",
                "A35",
                "A36",
                "A37",
                "A38",
                "A39",
            ],
            "registration_policy": {
                "a32_scope_limited_confirmation_required": True,
                "a32_confirmed_record_required": True,
                "zero_contradictions_required": True,
                "exact_game_candidate_context_measurement_scope_required": True,
                "unresolved_candidates_excluded": True,
                "non_identifiable_candidates_excluded": True,
                "a33_does_not_reevaluate_truth": True,
                "legacy_a33_1_registry_not_mutated": True,
            },
        },
        "source_a32_5_summary": dict(source.get("summary", {}) or {}),
        "scoped_confirmed_mechanics": [entry.to_dict() for entry in entries],
        "excluded_candidate_audit": excluded,
        "summary": summary,
        "outcome_status": summary["outcome_status"],
        "status": "REGISTERED" if entries else "NO_ELIGIBLE_ENTRY",
        "truth_status": A33_2_TRUTH_STATUS,
        "source_stage": "A32.5",
        "registry_stage": "A33.2",
        "registry_validation_performed": True,
        "registration_performed": bool(entries),
        "scientific_review_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "a33_write_performed": True,
        "source_a32_5_mutated": False,
        "legacy_a33_1_registry_mutated": False,
        "support": sum(entry.confirmed_support for entry in entries),
        "support_origin": "A32.5_DECISION_RECORDS_ONLY",
        "unresolved_candidates_excluded": True,
        "non_identifiable_candidates_excluded": True,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
    }


def build_scoped_unknown_game_registry(
    decisions_payload: Mapping[str, Any],
    *,
    source_artifact: str = "",
) -> Tuple[ScopedUnknownGameRegistryEntry, ...]:
    """Extract only A32.5 handoffs that preserve the complete confirmed scope."""
    validate_a32_5_scoped_registry_source(decisions_payload)
    decisions_by_candidate = {
        str(row.get("candidate_id", "")): row
        for row in decisions_payload.get("revision_decisions", []) or []
        if isinstance(row, Mapping)
    }
    entries: List[ScopedUnknownGameRegistryEntry] = []
    for handoff in decisions_payload.get("a33_handoff_candidates", []) or []:
        candidate_id = str(handoff.get("candidate_id", ""))
        decision = decisions_by_candidate[candidate_id]
        decision_record = dict(decision.get("decision_record", {}) or {})
        spec = mechanic_prediction_spec(str(handoff.get("candidate_key", "")))
        action_args = handoff.get("action_args")
        entries.append(
            ScopedUnknownGameRegistryEntry(
                key=str(handoff.get("candidate_key", "")),
                candidate_id=candidate_id,
                game_id=str(handoff.get("game_id", "")),
                action=str(handoff.get("action", "")),
                action_args=(
                    dict(action_args) if isinstance(action_args, Mapping) else None
                ),
                mechanic_family=spec["mechanic_family"],
                predicted_metric=str(handoff.get("measurement", "")),
                confirmed_support=int(handoff.get("support", 0) or 0),
                contradictions=int(handoff.get("contradictions", 0) or 0),
                experiments_spent=int(handoff.get("experiments_spent", 0) or 0),
                budgets=tuple(int(value) for value in handoff.get("budgets", []) or []),
                context_snapshot_hashes=tuple(
                    str(value)
                    for value in handoff.get("context_snapshot_hashes", []) or []
                ),
                parameterized_control_variants=tuple(
                    dict(row)
                    for row in handoff.get("parameterized_control_variants", []) or []
                    if isinstance(row, Mapping)
                ),
                source_experiment_ids=tuple(
                    str(value)
                    for value in handoff.get("source_experiment_ids", []) or []
                ),
                scoped_claim=str(handoff.get("scoped_claim", "")),
                source_artifact=source_artifact,
                evidence_notes=(
                    "a32_5_scope_limited_confirmation_reused_without_reevaluation",
                    "four_exact_preregistered_paired_experiments",
                    "parameterized_controls_not_relabelled_as_distinct_actions",
                    "scope_locked_to_game_candidate_contexts_and_measurement",
                ),
            )
        )
        if decision_record.get("key") != entries[-1].key:
            raise ValueError("A32.5 decision record and A33 handoff key must align")
    return tuple(entries)


def build_excluded_candidate_audit(
    decisions_payload: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Expose every A32.5 candidate deliberately omitted from confirmed memory."""
    handoff_ids = {
        str(row.get("candidate_id", ""))
        for row in decisions_payload.get("a33_handoff_candidates", []) or []
        if isinstance(row, Mapping)
    }
    excluded: List[Dict[str, Any]] = []
    for row in decisions_payload.get("revision_decisions", []) or []:
        if not isinstance(row, Mapping):
            continue
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id in handoff_ids:
            continue
        record = dict(row.get("decision_record", {}) or {})
        decision = str(row.get("decision", ""))
        if decision == KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_CONTROL:
            reason = "NON_IDENTIFIABLE_UNRESOLVED_NOT_REGISTRY_ELIGIBLE"
        else:
            reason = "NON_CONFIRMED_A32_5_RECORD_NOT_REGISTRY_ELIGIBLE"
        excluded.append(
            {
                "candidate_id": candidate_id,
                "candidate_key": str(row.get("candidate_key", "")),
                "game_id": str(row.get("game_id", "")),
                "action": str(row.get("action", "")),
                "action_args": row.get("action_args"),
                "source_decision": decision,
                "decision_record_status": str(record.get("status", "")),
                "decision_record_support": int(record.get("support", 0) or 0),
                "exclusion_reason": reason,
                "registered": False,
                "counted_as_refutation": False,
            }
        )
    return excluded


def summarize_scoped_unknown_game_registry(
    entries: Sequence[ScopedUnknownGameRegistryEntry],
    excluded: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    scope_locked = sum(
        entry.scope_game_locked
        and entry.scope_candidate_locked
        and entry.scope_contexts_locked
        and entry.scope_measurement_locked
        for entry in entries
    )
    return {
        "source_candidates_consumed": len(entries) + len(excluded),
        "a32_5_handoff_candidates_consumed": len(entries),
        "scoped_confirmed_mechanics_registered": len(entries),
        "unresolved_candidates_excluded": sum(
            str(row.get("decision_record_status", "")) == "unresolved"
            for row in excluded
        ),
        "non_identifiable_candidates_excluded": sum(
            str(row.get("exclusion_reason", ""))
            == "NON_IDENTIFIABLE_UNRESOLVED_NOT_REGISTRY_ELIGIBLE"
            for row in excluded
        ),
        "confirmed_support_imported_from_a32_5": sum(
            entry.confirmed_support for entry in entries
        ),
        "experiments_spent_total": sum(entry.experiments_spent for entry in entries),
        "registered_contexts": sum(
            len(entry.context_snapshot_hashes) for entry in entries
        ),
        "registered_parameterized_control_variants": sum(
            len(entry.parameterized_control_variants) for entry in entries
        ),
        "scope_locked_entries": scope_locked,
        "a33_truth_reevaluations": 0,
        "a33_confirmations_performed": 0,
        "legacy_a33_1_registry_mutated": False,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
        "outcome_status": A33_2_ENTRY_ADDED if entries else A33_2_NO_ELIGIBLE_ENTRY,
    }


def validate_a32_5_scoped_registry_source(source: Mapping[str, Any]) -> None:
    """Reject any A32.5 payload that would broaden or invent registry truth."""
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != A32_5_SCHEMA_VERSION:
        raise ValueError("A32.5 schema version is not supported by A33.2")
    if str(source.get("outcome_status", "")) != (
        A32_5_MIXED_SCOPE_CONFIRMATION_AND_NON_IDENTIFIABILITY
    ):
        raise ValueError("A33.2 expects the mixed A32.5 scoped outcome")
    if str(source.get("status", "")) != "MIXED_CONFIRMED_AND_UNRESOLVED":
        raise ValueError("A33.2 expects confirmed and unresolved A32.5 records")
    if str(source.get("truth_status", "")) != (
        "SCOPED_A32_DECISION_WITHOUT_EXTERNAL_ORACLE"
    ):
        raise ValueError("A32.5 truth status is not eligible for A33.2")
    if not bool(source.get("scientific_review_performed", False)) or not bool(
        source.get("revision_performed", False)
    ):
        raise ValueError("A32.5 scientific revision must be complete")
    if not bool(source.get("confirmation_performed", False)):
        raise ValueError("A32.5 must provide a confirmation before A33.2")
    if bool(source.get("refutation_performed", False)):
        raise ValueError("A33.2 source cannot contain a refutation in this outcome")
    if not bool(source.get("a33_ready", False)):
        raise ValueError("A32.5 must explicitly mark an A33-ready handoff")
    if bool(source.get("a33_write_performed", False)):
        raise ValueError("A32.5 cannot pre-write the A33.2 registry")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("A32.5 wrong_confirmations must remain 0")
    if bool(source.get("parameterized_controls_counted_as_distinct_actions", False)):
        raise ValueError("A32.5 cannot relabel parameterized controls")
    if bool(
        source.get("sage_candidate_events_counted_as_support_before_a32_review", False)
    ):
        raise ValueError("A32.5 cannot inherit pre-counted SAGE support")
    if bool(source.get("neutral_events_counted_as_refutation", False)) or bool(
        source.get("non_discrimination_counted_as_refutation", False)
    ):
        raise ValueError("A32.5 non-discrimination cannot count as refutation")
    if bool(source.get("scope_limited_confirmation_generalized_beyond_game", False)):
        raise ValueError("A32.5 confirmation cannot be generalized beyond its game")

    decisions = [
        row
        for row in source.get("revision_decisions", []) or []
        if isinstance(row, Mapping)
    ]
    handoffs = [
        row
        for row in source.get("a33_handoff_candidates", []) or []
        if isinstance(row, Mapping)
    ]
    if not decisions or not handoffs:
        raise ValueError("A33.2 requires A32.5 decisions and handoffs")
    decision_ids = [str(row.get("candidate_id", "")) for row in decisions]
    handoff_ids = [str(row.get("candidate_id", "")) for row in handoffs]
    if "" in decision_ids or len(decision_ids) != len(set(decision_ids)):
        raise ValueError("A32.5 decision candidate ids must be unique")
    if "" in handoff_ids or len(handoff_ids) != len(set(handoff_ids)):
        raise ValueError("A32.5 handoff candidate ids must be unique")
    if not set(handoff_ids).issubset(set(decision_ids)):
        raise ValueError("A32.5 handoffs must align with revision decisions")
    if len(decisions) != int(summary.get("scientific_revision_decisions", 0) or 0):
        raise ValueError("A32.5 revision decision count must be exact")
    if len(handoffs) != int(summary.get("a33_ready_candidates", 0) or 0):
        raise ValueError("A32.5 A33-ready candidate count must be exact")

    decisions_by_id = {
        str(row.get("candidate_id", "")): row for row in decisions
    }
    handoff_support = 0
    handoff_experiments: set[str] = set()
    handoff_keys: set[str] = set()
    for handoff in handoffs:
        candidate_id = str(handoff.get("candidate_id", ""))
        decision = decisions_by_id[candidate_id]
        record = dict(decision.get("decision_record", {}) or {})
        scope = dict(decision.get("scope_limits", {}) or {})
        key = str(handoff.get("candidate_key", ""))
        if str(decision.get("decision", "")) != (
            CONFIRM_SCOPE_LIMITED_AFTER_PARAMETERIZED_CONTROL_REVISION
        ):
            raise ValueError("A33.2 accepts only A32.5 scope-limited confirmations")
        if str(record.get("status", "")) != "confirmed":
            raise ValueError("A33.2 accepts only confirmed A32.5 records")
        if not bool(decision.get("confirmation_performed", False)) or not bool(
            decision.get("a33_ready", False)
        ):
            raise ValueError("A32.5 confirmed decision must be A33-ready")
        if bool(decision.get("refutation_performed", False)) or bool(
            decision.get("a33_write_performed", False)
        ):
            raise ValueError("A32.5 confirmed decision cannot refute or pre-write A33")
        if not key or key in handoff_keys:
            raise ValueError("A32.5 handoff keys must be non-empty and unique")
        if str(record.get("key", "")) != key or str(
            decision.get("candidate_key", "")
        ) != key:
            raise ValueError("A32.5 handoff and confirmed record keys must align")
        if str(handoff.get("status", "")) != "confirmed":
            raise ValueError("A32.5 handoff status must be confirmed")
        support = int(handoff.get("support", 0) or 0)
        contradictions = int(handoff.get("contradictions", 0) or 0)
        spent = int(handoff.get("experiments_spent", 0) or 0)
        if support <= 0 or support != int(record.get("support", 0) or 0):
            raise ValueError("A32.5 handoff support must match its confirmed record")
        if contradictions != 0 or contradictions != int(
            record.get("contradictions", 0) or 0
        ):
            raise ValueError("A33.2 requires zero A32.5 handoff contradictions")
        if spent <= 0 or spent != int(record.get("experiments_spent", 0) or 0):
            raise ValueError("A32.5 handoff experiment count must match its record")
        if not bool(handoff.get("ready_for_A33_registry_review", False)):
            raise ValueError("A32.5 handoff must be ready for registry review")
        if bool(handoff.get("a33_write_performed", False)):
            raise ValueError("A32.5 handoff cannot pre-write A33")
        if not bool(handoff.get("not_generalized_beyond_game", False)) or not bool(
            handoff.get("not_generalized_beyond_candidate_scope", False)
        ):
            raise ValueError("A32.5 handoff scope must remain locked")
        metric = str(handoff.get("measurement", ""))
        spec = mechanic_prediction_spec(key)
        if not metric or metric != spec["predicted_metric"]:
            raise ValueError("A32.5 handoff measurement must match its candidate key")
        if str(handoff.get("game_id", "")) != spec["game_id"] or str(
            handoff.get("action", "")
        ) != spec["action"]:
            raise ValueError("A32.5 handoff game and action must match its key")
        if (
            str(handoff.get("game_id", "")) != str(decision.get("game_id", ""))
            or str(handoff.get("action", "")) != str(decision.get("action", ""))
            or _canonical_json(handoff.get("action_args"))
            != _canonical_json(decision.get("action_args"))
        ):
            raise ValueError("A32.5 handoff identity must match its decision")
        contexts = [
            str(value)
            for value in handoff.get("context_snapshot_hashes", []) or []
        ]
        variants = [
            row
            for row in handoff.get("parameterized_control_variants", []) or []
            if isinstance(row, Mapping)
        ]
        experiment_ids = [
            str(value) for value in handoff.get("source_experiment_ids", []) or []
        ]
        budgets = [int(value) for value in handoff.get("budgets", []) or []]
        if not contexts or len(contexts) != len(set(contexts)):
            raise ValueError("A32.5 handoff contexts must be non-empty and unique")
        if len(variants) < 2:
            raise ValueError("A32.5 handoff needs at least two control variants")
        if len(experiment_ids) != spent or "" in experiment_ids:
            raise ValueError("A32.5 handoff experiment ids must match experiments spent")
        if (
            str(scope.get("game_id", "")) != str(handoff.get("game_id", ""))
            or str(scope.get("candidate_id", "")) != candidate_id
            or str(scope.get("action", "")) != str(handoff.get("action", ""))
            or _canonical_json(scope.get("action_args"))
            != _canonical_json(handoff.get("action_args"))
            or str(scope.get("measurement", "")) != metric
            or sorted(int(value) for value in scope.get("budgets", []) or [])
            != sorted(budgets)
            or sorted(str(value) for value in scope.get("context_snapshot_hashes", []) or [])
            != sorted(contexts)
        ):
            raise ValueError("A32.5 handoff must preserve its decision scope exactly")
        if experiment_ids != [
            str(value) for value in decision.get("source_experiment_ids", []) or []
        ]:
            raise ValueError("A32.5 handoff experiment ids must match its decision")
        if str(handoff.get("scoped_claim", "")) != str(
            record.get("description", "")
        ):
            raise ValueError("A32.5 handoff claim must match its confirmed record")
        if handoff_experiments.intersection(experiment_ids):
            raise ValueError("A32.5 handoff experiment ids cannot be reused")
        handoff_experiments.update(experiment_ids)
        handoff_keys.add(key)
        handoff_support += support

    if handoff_support != int(source.get("support", 0) or 0):
        raise ValueError("A32.5 top-level support must match A33 handoffs")
    if handoff_support != int(
        summary.get("scientific_support_counted_by_a32", 0) or 0
    ):
        raise ValueError("A32.5 summary support must match A33 handoffs")
    confirmed_count = sum(
        str(dict(row.get("decision_record", {}) or {}).get("status", ""))
        == "confirmed"
        for row in decisions
    )
    if confirmed_count != len(handoffs):
        raise ValueError("every confirmed A32.5 record must have one A33 handoff")


def write_scoped_unknown_game_registry(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the A33.2 scope-locked unknown-game registry."
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=DEFAULT_A32_UNKNOWN_GAME_PARAMETERIZED_CONTROL_REVISION_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_scoped_unknown_game_registry_generation(
        decisions_path=args.decisions
    )
    write_scoped_unknown_game_registry(payload, args.out)
    print(
        json.dumps(
            {"output_path": str(args.out), "summary": payload["summary"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
