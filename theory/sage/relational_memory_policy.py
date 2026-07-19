"""SAGE.8a scope-locked integration of A33.3 and A33.4 memories."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from theory.a33.control_dependent_relational_registry import (
    A33_3_ENTRY_ADDED,
    A33_3_SCHEMA_VERSION,
    A33_3_TRUTH_STATUS,
    DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH,
)
from theory.a33.parameterized_relational_registry import (
    A33_4_ENTRY_ADDED,
    A33_4_SCHEMA_VERSION,
    A33_4_TRUTH_STATUS,
    DEFAULT_A33_PARAMETERIZED_RELATIONAL_REGISTRY_OUTPUT_PATH,
)

from .policy_loop_guard import action_args, action_name


DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH = (
    Path("diagnostics") / "sage" / "sage8a_relational_memory_policy.json"
)

SAGE8A_SCHEMA_VERSION = "sage.relational_memory_policy.v1"
SAGE8A_TRUTH_STATUS = "NOT_REEVALUATED_BY_SAGE_8A"
SAGE8A_POLICY_READY = "SAGE_RELATIONAL_MEMORY_POLICY_READY_FOR_COMPARISON"
SAGE8A_POLICY_INCOMPLETE = "SAGE_RELATIONAL_MEMORY_POLICY_INCOMPLETE"

LOWER_EFFECT_COMPARATOR_MATCH = "LOWER_EFFECT_COMPARATOR_MATCH"
EQUIVALENT_COMPARATOR_PRESERVED = "EQUIVALENT_COMPARATOR_PRESERVED"
CONTEXT_OUT_OF_SCOPE = "CONTEXT_OUT_OF_SCOPE"
GAME_OUT_OF_SCOPE = "GAME_OUT_OF_SCOPE"
PROPOSED_ACTION_OUT_OF_SCOPE = "PROPOSED_ACTION_OUT_OF_SCOPE"
TARGET_UNAVAILABLE = "TARGET_UNAVAILABLE"


@dataclass(frozen=True)
class RelationalMemoryPolicyEntry:
    """One scope-locked action replacement rule derived from confirmed memory."""

    policy_entry_id: str
    registry_source: str
    registry_key: str
    registry_entry_type: str
    game_id: str
    metric: str
    context_snapshot_hashes: Tuple[str, ...]
    lower_effect_action: str
    lower_effect_action_args: Dict[str, Any]
    selected_action: str
    selected_action_args: Dict[str, Any]
    equivalent_action: str
    equivalent_action_args: Dict[str, Any]
    confirmed_support: int
    evidence_notes: Tuple[str, ...] = field(default_factory=tuple)
    scope_game_locked: bool = True
    scope_contexts_locked: bool = True
    scope_metric_locked: bool = True
    scope_action_identity_locked: bool = True
    autonomous_effect_not_assumed: bool = True
    truth_reevaluated_by_policy: bool = False
    support_recounted_by_policy: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_entry_id": self.policy_entry_id,
            "registry_source": self.registry_source,
            "registry_key": self.registry_key,
            "registry_entry_type": self.registry_entry_type,
            "game_id": self.game_id,
            "metric": self.metric,
            "context_snapshot_hashes": list(self.context_snapshot_hashes),
            "lower_effect_action": self.lower_effect_action,
            "lower_effect_action_args": dict(self.lower_effect_action_args),
            "selected_action": self.selected_action,
            "selected_action_args": dict(self.selected_action_args),
            "equivalent_action": self.equivalent_action,
            "equivalent_action_args": dict(self.equivalent_action_args),
            "confirmed_support": int(self.confirmed_support),
            "evidence_notes": list(self.evidence_notes),
            "scope_game_locked": self.scope_game_locked,
            "scope_contexts_locked": self.scope_contexts_locked,
            "scope_metric_locked": self.scope_metric_locked,
            "scope_action_identity_locked": self.scope_action_identity_locked,
            "autonomous_effect_not_assumed": self.autonomous_effect_not_assumed,
            "truth_reevaluated_by_policy": self.truth_reevaluated_by_policy,
            "support_recounted_by_policy": self.support_recounted_by_policy,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


@dataclass(frozen=True)
class PolicyActionOption:
    """Minimal live-action shape used for deterministic policy audits."""

    name: str
    action_args: Dict[str, Any] = field(default_factory=dict)


def run_sage8a_relational_memory_policy_integration(
    *,
    a33_3_registry_path: str | Path = (
        DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH
    ),
    a33_4_registry_path: str | Path = (
        DEFAULT_A33_PARAMETERIZED_RELATIONAL_REGISTRY_OUTPUT_PATH
    ),
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Compile confirmed relations into an executable, scope-safe SAGE policy."""
    a33_3 = _load_json(a33_3_registry_path)
    a33_4 = _load_json(a33_4_registry_path)
    entries = build_relational_memory_policy_entries(a33_3, a33_4)
    policy = build_relational_memory_policy_payload(entries)
    audit = build_relational_memory_policy_audit(policy, entries)
    gate = build_sage8a_gate(entries, audit)
    if not gate or not all(gate.values()):
        raise ValueError("SAGE.8a relational memory policy gate did not pass")
    summary = summarize_sage8a(entries, audit, gate)
    payload = {
        "config": {
            "schema_version": SAGE8A_SCHEMA_VERSION,
            "a33_3_registry_path": str(a33_3_registry_path),
            "a33_4_registry_path": str(a33_4_registry_path),
            "inputs_read": ["A33.3", "A33.4"],
            "artifacts_not_modified": ["A33.1", "A33.2", "A33.3", "A33.4"],
            "application_policy": {
                "exact_game_match_required": True,
                "exact_context_hash_match_required": True,
                "exact_lower_effect_comparator_match_required": True,
                "selected_action_must_be_live_legal": True,
                "equivalent_comparator_is_never_overridden": True,
                "autonomous_effect_is_never_assumed": True,
                "out_of_scope_falls_back_unchanged": True,
                "registry_truth_is_not_reevaluated": True,
                "registry_support_is_not_recounted": True,
            },
        },
        "policy": policy,
        "policy_entries": [entry.to_dict() for entry in entries],
        "integration_audit": audit,
        "gate": gate,
        "summary": summary,
        "outcome_status": summary["outcome_status"],
        "status": "READY" if summary["gate_passed"] else "INCOMPLETE",
        "truth_status": SAGE8A_TRUTH_STATUS,
        "policy_integration_performed": True,
        "comparative_evaluation_performed": False,
        "scientific_review_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "support": 0,
        "registry_support_recounted": False,
        "a33_mutated": False,
        "scope_generalization_performed": False,
        "ready_for_comparative_evaluation": True,
        "wrong_confirmations": 0,
    }
    if output_path is not None:
        write_sage8a_relational_memory_policy(payload, output_path)
    return payload


def build_relational_memory_policy_entries(
    a33_3: Mapping[str, Any],
    a33_4: Mapping[str, Any],
) -> Tuple[RelationalMemoryPolicyEntry, ...]:
    validate_relational_memory_registry_sources(a33_3, a33_4)
    wa30 = dict(a33_3["control_dependent_relational_contrasts"][0])
    tn36 = dict(a33_4["parameterized_relational_contrasts"][0])
    entries = (
        RelationalMemoryPolicyEntry(
            policy_entry_id="sage8a::relational_memory::wa30::001",
            registry_source="A33.3",
            registry_key=str(wa30.get("key", "")),
            registry_entry_type=str(wa30.get("registry_entry_type", "")),
            game_id=str(wa30.get("game_id", "")),
            metric=str(wa30.get("predicted_metric", "")),
            context_snapshot_hashes=tuple(
                str(value)
                for value in wa30.get("paired_context_snapshot_hashes", []) or []
            ),
            lower_effect_action="ACTION1",
            lower_effect_action_args={},
            selected_action=str(wa30.get("target_action", "")),
            selected_action_args={},
            equivalent_action="ACTION3",
            equivalent_action_args={},
            confirmed_support=int(wa30.get("confirmed_support", 0) or 0),
            evidence_notes=(
                "replace_only_recorded_action1_lower_effect_comparator",
                "preserve_action3_equivalent_comparator",
                "apply_only_in_three_exact_wa30_context_hashes",
            ),
        ),
        RelationalMemoryPolicyEntry(
            policy_entry_id="sage8a::relational_memory::tn36::001",
            registry_source="A33.4",
            registry_key=str(tn36.get("key", "")),
            registry_entry_type=str(tn36.get("registry_entry_type", "")),
            game_id=str(tn36.get("game_id", "")),
            metric=str(tn36.get("predicted_metric", "")),
            context_snapshot_hashes=tuple(
                str(value) for value in tn36.get("context_snapshot_hashes", []) or []
            ),
            lower_effect_action="ACTION6",
            lower_effect_action_args={"x": 34, "y": 51},
            selected_action=str(tn36.get("target_action", "")),
            selected_action_args=dict(tn36.get("target_action_args", {}) or {}),
            equivalent_action="ACTION6",
            equivalent_action_args={"x": 41, "y": 44},
            confirmed_support=int(tn36.get("confirmed_support", 0) or 0),
            evidence_notes=(
                "replace_only_recorded_x34_y51_lower_effect_variant",
                "preserve_x41_y44_equivalent_variant",
                "apply_only_in_eight_exact_tn36_context_hashes",
                "parameter_variants_remain_one_action_family",
            ),
        ),
    )
    return entries


def build_relational_memory_policy_payload(
    entries: Sequence[RelationalMemoryPolicyEntry],
) -> Dict[str, Any]:
    return {
        "policy_id": "sage8a::scope_locked_relational_memory_policy",
        "policy_version": 1,
        "enabled": True,
        "application_mode": "EXACT_SCOPE_LOWER_EFFECT_COMPARATOR_REPLACEMENT",
        "entries": [entry.to_dict() for entry in entries],
        "fallback_policy": "PRESERVE_PROPOSED_ACTION_UNCHANGED",
        "truth_status": SAGE8A_TRUTH_STATUS,
        "support": 0,
        "parameterized_variants_counted_as_distinct_actions": False,
        "wrong_confirmations": 0,
    }


def apply_relational_memory_policy(
    policy: Mapping[str, Any],
    *,
    game_id: str,
    context_snapshot_hash: str,
    proposed_action_raw: Any,
    valid_actions: Sequence[Any],
    metric: str = "local_patch_before_after",
) -> Dict[str, Any]:
    """Replace only an exact registered lower-effect comparator."""
    proposed_name = action_name(proposed_action_raw)
    proposed_args = action_args(proposed_action_raw)
    base = {
        "selected_action_raw": proposed_action_raw,
        "selected_action": proposed_name,
        "selected_action_args": dict(proposed_args),
        "relational_memory_consulted": True,
        "relational_memory_applied": False,
        "policy_entry_id": "",
        "registry_source": "",
        "decision_reason": PROPOSED_ACTION_OUT_OF_SCOPE,
        "scope_match": False,
        "game_scope_match": False,
        "context_scope_match": False,
        "metric_scope_match": False,
        "proposed_lower_effect_comparator_match": False,
        "equivalent_comparator_preserved": False,
        "selected_action_live_legal": proposed_action_raw is not None,
        "truth_reevaluated": False,
        "support_counted": 0,
        "wrong_confirmations": 0,
    }
    entries = [
        row for row in policy.get("entries", []) or [] if isinstance(row, Mapping)
    ]
    game_entries = [row for row in entries if str(row.get("game_id", "")) == game_id]
    if not game_entries:
        base["decision_reason"] = GAME_OUT_OF_SCOPE
        return base
    context_entries = [
        row
        for row in game_entries
        if context_snapshot_hash in set(row.get("context_snapshot_hashes", []) or [])
    ]
    if not context_entries:
        base.update({"game_scope_match": True, "decision_reason": CONTEXT_OUT_OF_SCOPE})
        return base
    entry = context_entries[0]
    metric_match = str(entry.get("metric", "")) == metric
    equivalent_match = _action_identity_matches(
        proposed_name,
        proposed_args,
        str(entry.get("equivalent_action", "")),
        dict(entry.get("equivalent_action_args", {}) or {}),
    )
    lower_match = _action_identity_matches(
        proposed_name,
        proposed_args,
        str(entry.get("lower_effect_action", "")),
        dict(entry.get("lower_effect_action_args", {}) or {}),
    )
    base.update(
        {
            "policy_entry_id": str(entry.get("policy_entry_id", "")),
            "registry_source": str(entry.get("registry_source", "")),
            "game_scope_match": True,
            "context_scope_match": True,
            "metric_scope_match": metric_match,
            "scope_match": metric_match,
            "proposed_lower_effect_comparator_match": lower_match,
            "equivalent_comparator_preserved": equivalent_match,
        }
    )
    if equivalent_match:
        base["decision_reason"] = EQUIVALENT_COMPARATOR_PRESERVED
        return base
    if not metric_match or not lower_match:
        base["decision_reason"] = PROPOSED_ACTION_OUT_OF_SCOPE
        return base
    selected = _select_action_option(
        valid_actions,
        str(entry.get("selected_action", "")),
        dict(entry.get("selected_action_args", {}) or {}),
    )
    if selected is None:
        base.update(
            {
                "decision_reason": TARGET_UNAVAILABLE,
                "selected_action_live_legal": False,
            }
        )
        return base
    base.update(
        {
            "selected_action_raw": selected,
            "selected_action": action_name(selected),
            "selected_action_args": action_args(selected),
            "relational_memory_applied": True,
            "decision_reason": LOWER_EFFECT_COMPARATOR_MATCH,
            "selected_action_live_legal": True,
        }
    )
    return base


def build_relational_memory_policy_audit(
    policy: Mapping[str, Any],
    entries: Sequence[RelationalMemoryPolicyEntry],
) -> List[Dict[str, Any]]:
    """Exercise apply/preserve/quarantine branches for every compiled entry."""
    audit: List[Dict[str, Any]] = []
    for entry in entries:
        options = (
            PolicyActionOption(
                entry.lower_effect_action, entry.lower_effect_action_args
            ),
            PolicyActionOption(entry.selected_action, entry.selected_action_args),
            PolicyActionOption(entry.equivalent_action, entry.equivalent_action_args),
        )
        context_hash = entry.context_snapshot_hashes[0]
        applied = apply_relational_memory_policy(
            policy,
            game_id=entry.game_id,
            context_snapshot_hash=context_hash,
            proposed_action_raw=options[0],
            valid_actions=options,
            metric=entry.metric,
        )
        equivalent = apply_relational_memory_policy(
            policy,
            game_id=entry.game_id,
            context_snapshot_hash=context_hash,
            proposed_action_raw=options[2],
            valid_actions=options,
            metric=entry.metric,
        )
        wrong_context = apply_relational_memory_policy(
            policy,
            game_id=entry.game_id,
            context_snapshot_hash="out_of_scope_hash",
            proposed_action_raw=options[0],
            valid_actions=options,
            metric=entry.metric,
        )
        wrong_game = apply_relational_memory_policy(
            policy,
            game_id="other-game",
            context_snapshot_hash=context_hash,
            proposed_action_raw=options[0],
            valid_actions=options,
            metric=entry.metric,
        )
        audit.append(
            {
                "policy_entry_id": entry.policy_entry_id,
                "registry_source": entry.registry_source,
                "game_id": entry.game_id,
                "exact_context_application": _serializable_decision(applied),
                "equivalent_comparator_audit": _serializable_decision(equivalent),
                "wrong_context_audit": _serializable_decision(wrong_context),
                "wrong_game_audit": _serializable_decision(wrong_game),
            }
        )
    return audit


def build_sage8a_gate(
    entries: Sequence[RelationalMemoryPolicyEntry],
    audit: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    return {
        "both_relational_registries_compiled_once": len(entries) == len(audit) == 2,
        "wa30_and_tn36_scopes_present": {entry.game_id for entry in entries}
        == {"wa30-ee6fef47", "tn36-ab4f63cc"},
        "all_exact_lower_comparators_replaced": all(
            bool(
                row.get("exact_context_application", {}).get(
                    "relational_memory_applied", False
                )
            )
            and str(row.get("exact_context_application", {}).get("decision_reason", ""))
            == LOWER_EFFECT_COMPARATOR_MATCH
            for row in audit
        ),
        "all_equivalent_comparators_preserved": all(
            not bool(
                row.get("equivalent_comparator_audit", {}).get(
                    "relational_memory_applied", True
                )
            )
            and str(
                row.get("equivalent_comparator_audit", {}).get("decision_reason", "")
            )
            == EQUIVALENT_COMPARATOR_PRESERVED
            for row in audit
        ),
        "wrong_contexts_fall_back_unchanged": all(
            not bool(
                row.get("wrong_context_audit", {}).get(
                    "relational_memory_applied", True
                )
            )
            and str(row.get("wrong_context_audit", {}).get("decision_reason", ""))
            == CONTEXT_OUT_OF_SCOPE
            for row in audit
        ),
        "wrong_games_fall_back_unchanged": all(
            not bool(
                row.get("wrong_game_audit", {}).get("relational_memory_applied", True)
            )
            and str(row.get("wrong_game_audit", {}).get("decision_reason", ""))
            == GAME_OUT_OF_SCOPE
            for row in audit
        ),
        "policy_does_not_reevaluate_or_recount": all(
            not entry.truth_reevaluated_by_policy
            and not entry.support_recounted_by_policy
            for entry in entries
        ),
        "wrong_confirmations_zero": all(
            entry.wrong_confirmations == 0 for entry in entries
        ),
    }


def summarize_sage8a(
    entries: Sequence[RelationalMemoryPolicyEntry],
    audit: Sequence[Mapping[str, Any]],
    gate: Mapping[str, bool],
) -> Dict[str, Any]:
    gate_passed = bool(gate) and all(bool(value) for value in gate.values())
    return {
        "registry_entries_consumed": len(entries),
        "policy_entries_compiled": len(entries),
        "games_scoped": sorted({entry.game_id for entry in entries}),
        "exact_context_hashes_scoped": sum(
            len(entry.context_snapshot_hashes) for entry in entries
        ),
        "exact_application_audits": sum(
            bool(
                row.get("exact_context_application", {}).get(
                    "relational_memory_applied", False
                )
            )
            for row in audit
        ),
        "equivalent_comparators_preserved": sum(
            bool(
                row.get("equivalent_comparator_audit", {}).get(
                    "equivalent_comparator_preserved", False
                )
            )
            for row in audit
        ),
        "wrong_context_overrides": sum(
            bool(
                row.get("wrong_context_audit", {}).get(
                    "relational_memory_applied", False
                )
            )
            for row in audit
        ),
        "wrong_game_overrides": sum(
            bool(
                row.get("wrong_game_audit", {}).get("relational_memory_applied", False)
            )
            for row in audit
        ),
        "autonomous_effects_assumed": 0,
        "registry_truth_reevaluations": 0,
        "registry_support_recounted": 0,
        "comparative_evaluation_performed": False,
        "ready_for_comparative_evaluation": gate_passed,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
        "gate_passed": gate_passed,
        "outcome_status": SAGE8A_POLICY_READY
        if gate_passed
        else SAGE8A_POLICY_INCOMPLETE,
    }


def validate_relational_memory_registry_sources(
    a33_3: Mapping[str, Any],
    a33_4: Mapping[str, Any],
) -> None:
    config3 = dict(a33_3.get("config", {}) or {})
    config4 = dict(a33_4.get("config", {}) or {})
    if (
        str(config3.get("schema_version", "")) != A33_3_SCHEMA_VERSION
        or str(a33_3.get("outcome_status", "")) != A33_3_ENTRY_ADDED
        or str(a33_3.get("truth_status", "")) != A33_3_TRUTH_STATUS
        or str(a33_3.get("status", "")) != "REGISTERED"
        or bool(a33_3.get("confirmation_performed", True))
        or int(a33_3.get("wrong_confirmations", 0) or 0) != 0
    ):
        raise ValueError("SAGE.8a requires the completed A33.3 registry")
    if (
        str(config4.get("schema_version", "")) != A33_4_SCHEMA_VERSION
        or str(a33_4.get("outcome_status", "")) != A33_4_ENTRY_ADDED
        or str(a33_4.get("truth_status", "")) != A33_4_TRUTH_STATUS
        or str(a33_4.get("status", "")) != "REGISTERED"
        or bool(a33_4.get("confirmation_performed", True))
        or int(a33_4.get("wrong_confirmations", 0) or 0) != 0
    ):
        raise ValueError("SAGE.8a requires the completed A33.4 registry")
    if not bool(a33_3.get("summary", {}).get("gate_passed", False)) or not all(
        bool(value) for value in a33_3.get("gate", {}).values()
    ):
        raise ValueError("every A33.3 registry gate must pass")
    if not bool(a33_4.get("summary", {}).get("gate_passed", False)) or not all(
        bool(value) for value in a33_4.get("gate", {}).values()
    ):
        raise ValueError("every A33.4 registry gate must pass")
    entries3 = list(a33_3.get("control_dependent_relational_contrasts", []) or [])
    entries4 = list(a33_4.get("parameterized_relational_contrasts", []) or [])
    if len(entries3) != 1 or len(entries4) != 1:
        raise ValueError("SAGE.8a requires exactly one entry from each registry")
    wa30 = entries3[0]
    tn36 = entries4[0]
    if (
        str(wa30.get("game_id", "")) != "wa30-ee6fef47"
        or str(wa30.get("target_action", "")) != "ACTION2"
        or list(wa30.get("control_actions", []) or []) != ["ACTION1", "ACTION3"]
        or len(wa30.get("paired_context_snapshot_hashes", []) or []) != 3
        or not bool(wa30.get("standalone_action2_effect_excluded", False))
    ):
        raise ValueError("SAGE.8a requires the exact scope-locked A33.3 relation")
    if (
        str(tn36.get("game_id", "")) != "tn36-ab4f63cc"
        or str(tn36.get("target_action", "")) != "ACTION6"
        or dict(tn36.get("target_action_args", {}) or {}) != {"x": 25, "y": 42}
        or len(tn36.get("context_snapshot_hashes", []) or []) != 8
        or not bool(tn36.get("autonomous_target_effect_excluded", False))
    ):
        raise ValueError("SAGE.8a requires the exact scope-locked A33.4 relation")


def write_sage8a_relational_memory_policy(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _select_action_option(
    valid_actions: Sequence[Any],
    expected_name: str,
    expected_args: Mapping[str, Any],
) -> Any | None:
    for option in valid_actions:
        if _action_identity_matches(
            action_name(option), action_args(option), expected_name, expected_args
        ):
            return option
    return None


def _action_identity_matches(
    actual_name: str,
    actual_args: Mapping[str, Any],
    expected_name: str,
    expected_args: Mapping[str, Any],
) -> bool:
    return str(actual_name) == str(expected_name) and dict(actual_args or {}) == dict(
        expected_args or {}
    )


def _serializable_decision(decision: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in decision.items() if key != "selected_action_raw"
    }


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a33-3",
        default=str(DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH),
    )
    parser.add_argument(
        "--a33-4",
        default=str(DEFAULT_A33_PARAMETERIZED_RELATIONAL_REGISTRY_OUTPUT_PATH),
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH)
    )
    args = parser.parse_args(argv)
    payload = run_sage8a_relational_memory_policy_integration(
        a33_3_registry_path=args.a33_3,
        a33_4_registry_path=args.a33_4,
        output_path=args.out,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
