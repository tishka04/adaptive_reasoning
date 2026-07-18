"""SAGE.6 bounded transfer to a deterministically selected second unknown game."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import game_splits

from theory.a33.scoped_unknown_game_registry import (
    A33_2_ENTRY_ADDED,
    A33_2_SCHEMA_VERSION,
    A33_2_TRUTH_STATUS,
    DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH,
)

from .long_horizon_transfer import DEFAULT_BUDGETS
from .unknown_game_bounded_probe import (
    DEFAULT_HUMAN_TRACES_DIR,
    DEFAULT_M2_ARC_LEWM_DATASET_MANIFEST_PATH,
    DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH,
    SAGE5_ALL_BUDGETS_PASSED,
    SAGE5_SCHEMA_VERSION,
    SAGE5_TRUTH_STATUS,
    EnvFactory,
    run_sage5_unknown_game_bounded_probe,
    unknown_game_identity,
)


DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH = (
    Path("diagnostics") / "sage" / "sage6_second_unknown_game_transfer_results.json"
)

SAGE6_SCHEMA_VERSION = "sage.second_unknown_game_transfer.v1"
SAGE6_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_6"
SAGE6_ALL_BUDGETS_PASSED = (
    "SAGE_SECOND_UNKNOWN_GAME_ALL_BUDGETS_GATE_PASSED_CANDIDATE_ONLY"
)
SAGE6_TRANSFER_GATE_FAILED = (
    "SAGE_SECOND_UNKNOWN_GAME_TRANSFER_GATE_FAILED_CANDIDATE_ONLY"
)

ELIGIBLE_UNKNOWN_GAME = "ELIGIBLE_UNKNOWN_GAME"
EXCLUDED_SOURCE_GAME = "EXCLUDED_SOURCE_GAME"
EXCLUDED_SCOPE_LOCKED_REGISTRY_GAME = "EXCLUDED_SCOPE_LOCKED_REGISTRY_GAME"
EXCLUDED_KNOWN_GAME = "EXCLUDED_KNOWN_GAME"

DEFAULT_SELECTION_POLICY = "PUBLIC_UNSEEN_FIXED_ORDER_BEFORE_EXECUTION"


def run_sage6_second_unknown_game_transfer(
    *,
    source_sage5_path: str | Path = DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH,
    source_a33_2_path: str | Path = (
        DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH
    ),
    m2_dataset_manifest_path: str | Path = (
        DEFAULT_M2_ARC_LEWM_DATASET_MANIFEST_PATH
    ),
    human_traces_dir: str | Path = DEFAULT_HUMAN_TRACES_DIR,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    candidate_game_ids: Sequence[str] | None = None,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    seed: int = 0,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Run the SAGE.5 discipline on a second, preselected unknown game."""
    source_sage5 = _load_json(source_sage5_path)
    source_a33_2 = _load_json(source_a33_2_path)
    validate_sage5_transfer_source(source_sage5)
    validate_a33_2_transfer_source(source_a33_2, source_sage5=source_sage5)

    source_game_id = str(source_sage5.get("config", {}).get("game_id", ""))
    registry_game_ids = tuple(
        sorted(
            {
                str(row.get("game_id", ""))
                for row in source_a33_2.get("scoped_confirmed_mechanics", []) or []
                if isinstance(row, Mapping) and str(row.get("game_id", ""))
            }
        )
    )
    manifest = _load_optional_json(m2_dataset_manifest_path)
    requested_candidates = tuple(
        candidate_game_ids
        if candidate_game_ids is not None
        else (
            game_splits.resolve_full_game_id(short_id)
            for short_id in game_splits.PUBLIC_UNSEEN
        )
    )
    selection_audit, selected_game_id = select_second_unknown_game(
        source_game_id=source_game_id,
        candidate_game_ids=requested_candidates,
        registry_game_ids=registry_game_ids,
        human_traces_dir=human_traces_dir,
        m2_dataset_manifest=manifest,
    )

    second_probe = run_sage5_unknown_game_bounded_probe(
        m2_dataset_manifest_path=m2_dataset_manifest_path,
        human_traces_dir=human_traces_dir,
        environments_dir=environments_dir,
        output_path=None,
        game_id=selected_game_id,
        budgets=budgets,
        seed=seed,
        env_factory=env_factory,
    )
    validate_sage5_transfer_source(second_probe)
    if _same_game(selected_game_id, source_game_id):
        raise ValueError("SAGE.6 selected game must differ from the SAGE.5 game")
    if any(_same_game(selected_game_id, game_id) for game_id in registry_game_ids):
        raise ValueError("SAGE.6 selected game cannot reuse a scope-locked registry game")

    transfer_guard = build_cross_game_transfer_guard(
        source_game_id=source_game_id,
        selected_game_id=selected_game_id,
        registry_game_ids=registry_game_ids,
        source_a33_2=source_a33_2,
    )
    summary = summarize_sage6_transfer(
        source_sage5=source_sage5,
        source_a33_2=source_a33_2,
        selection_audit=selection_audit,
        selected_game_id=selected_game_id,
        second_probe=second_probe,
        transfer_guard=transfer_guard,
    )
    gate = build_sage6_gate(
        source_sage5=source_sage5,
        source_a33_2=source_a33_2,
        selected_game_id=selected_game_id,
        second_probe=second_probe,
        transfer_guard=transfer_guard,
    )
    outcome = (
        SAGE6_ALL_BUDGETS_PASSED
        if all(bool(value) for value in gate.values())
        else SAGE6_TRANSFER_GATE_FAILED
    )
    summary["gate_passed"] = outcome == SAGE6_ALL_BUDGETS_PASSED
    summary["outcome_status"] = outcome

    payload = {
        "config": {
            "schema_version": SAGE6_SCHEMA_VERSION,
            "source_sage5_path": str(source_sage5_path),
            "source_a33_2_path": str(source_a33_2_path),
            "m2_dataset_manifest_path": str(m2_dataset_manifest_path),
            "human_traces_dir": str(human_traces_dir),
            "environments_dir": (
                str(environments_dir) if environments_dir is not None else None
            ),
            "candidate_game_ids_in_fixed_order": list(requested_candidates),
            "selection_policy": DEFAULT_SELECTION_POLICY,
            "selection_occurs_before_second_game_execution": True,
            "outcome_based_game_selection_allowed": False,
            "budgets": [int(value) for value in budgets],
            "seed": int(seed),
            "inputs_read": ["SAGE.5", "A33.2", "M2 manifest"],
            "artifacts_not_modified": [
                "SAGE.5",
                "A32.5",
                "A33.1",
                "A33.2",
                "M2",
                "M3",
                "A40",
                "P2",
            ],
            "transfer_policy": {
                "second_game_must_differ_from_source_game": True,
                "unknown_game_hygiene_required": True,
                "source_scoped_mechanics_are_quarantined": True,
                "cross_game_mechanic_reuse_allowed": False,
                "policy_results_remain_candidate_only": True,
                "scientific_support_remains_zero": True,
            },
        },
        "source_loop_context": {
            "source_game_id": source_game_id,
            "source_sage5_outcome_status": str(source_sage5.get("outcome_status", "")),
            "source_sage5_all_budgets_gate_passed": bool(
                source_sage5.get("summary", {}).get("all_budgets_gate_passed", False)
            ),
            "source_a33_2_outcome_status": str(source_a33_2.get("outcome_status", "")),
            "source_a33_2_registration_performed": bool(
                source_a33_2.get("registration_performed", False)
            ),
            "source_scoped_registry_game_ids": list(registry_game_ids),
            "source_loop_closed_before_transfer": True,
        },
        "selection_audit": selection_audit,
        "selected_second_unknown_game": {
            **dict(second_probe.get("unknown_game_identity", {}) or {}),
            "source_probe_truth_status": str(
                second_probe.get("unknown_game_identity", {}).get("truth_status", "")
            ),
            "truth_status": SAGE6_TRUTH_STATUS,
            "selected_by_policy": DEFAULT_SELECTION_POLICY,
            "selection_rank": next(
                int(row.get("eligible_rank", 0) or 0)
                for row in selection_audit
                if bool(row.get("selected", False))
            ),
            "selected_before_execution": True,
            "selected_from_outcome_metrics": False,
        },
        "cross_game_transfer_guard": transfer_guard,
        "second_game_probe": second_probe,
        "gate": gate,
        "summary": summary,
        "status": "UNRESOLVED",
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE6_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_confirmation": False,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "scope_generalization_performed": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage6_second_unknown_game_transfer(payload, output_path)
    return payload


def select_second_unknown_game(
    *,
    source_game_id: str,
    candidate_game_ids: Sequence[str],
    registry_game_ids: Sequence[str],
    human_traces_dir: str | Path,
    m2_dataset_manifest: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    """Select the first eligible candidate in a fixed order, before execution."""
    audit: List[Dict[str, Any]] = []
    eligible_rank = 0
    for candidate_index, candidate in enumerate(candidate_game_ids, start=1):
        game_id = game_splits.resolve_full_game_id(str(candidate))
        identity = unknown_game_identity(
            game_id=game_id,
            human_traces_dir=human_traces_dir,
            m2_dataset_manifest=m2_dataset_manifest,
        )
        if _same_game(game_id, source_game_id):
            eligibility_status = EXCLUDED_SOURCE_GAME
        elif any(_same_game(game_id, scoped) for scoped in registry_game_ids):
            eligibility_status = EXCLUDED_SCOPE_LOCKED_REGISTRY_GAME
        elif not bool(identity.get("unknown_game", False)):
            eligibility_status = EXCLUDED_KNOWN_GAME
        else:
            eligibility_status = ELIGIBLE_UNKNOWN_GAME
            eligible_rank += 1
        audit.append(
            {
                "candidate_index": candidate_index,
                "game_id": game_id,
                "short_game_id": str(identity.get("short_game_id", "")),
                "split": str(identity.get("split", "")),
                "unknown_game": bool(identity.get("unknown_game", False)),
                "no_human_trace_for_game": bool(
                    identity.get("no_human_trace_for_game", False)
                ),
                "no_m2_arc_lewm_trace_for_game": bool(
                    identity.get("no_m2_arc_lewm_trace_for_game", False)
                ),
                "no_game_specific_prior": bool(
                    identity.get("no_game_specific_prior", False)
                ),
                "eligibility_status": eligibility_status,
                "eligible_rank": (
                    eligible_rank if eligibility_status == ELIGIBLE_UNKNOWN_GAME else 0
                ),
                "selected": False,
                "outcome_metrics_read_for_selection": False,
            }
        )
    selected = next(
        (row for row in audit if row["eligibility_status"] == ELIGIBLE_UNKNOWN_GAME),
        None,
    )
    if selected is None:
        raise ValueError("SAGE.6 found no eligible second unknown game")
    selected["selected"] = True
    return audit, str(selected["game_id"])


def build_cross_game_transfer_guard(
    *,
    source_game_id: str,
    selected_game_id: str,
    registry_game_ids: Sequence[str],
    source_a33_2: Mapping[str, Any],
) -> Dict[str, Any]:
    entries = [
        row
        for row in source_a33_2.get("scoped_confirmed_mechanics", []) or []
        if isinstance(row, Mapping)
    ]
    return {
        "source_game_id": source_game_id,
        "selected_game_id": selected_game_id,
        "different_game": not _same_game(source_game_id, selected_game_id),
        "scope_locked_registry_entries_available": len(entries),
        "scope_locked_registry_game_ids": list(registry_game_ids),
        "selected_game_outside_registry_scopes": not any(
            _same_game(selected_game_id, game_id) for game_id in registry_game_ids
        ),
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "source_action5_prior_applied": False,
        "source_action6_candidate_applied": False,
        "scope_generalization_performed": False,
        "registry_read_only": True,
        "quarantine_passed": True,
    }


def build_sage6_gate(
    *,
    source_sage5: Mapping[str, Any],
    source_a33_2: Mapping[str, Any],
    selected_game_id: str,
    second_probe: Mapping[str, Any],
    transfer_guard: Mapping[str, Any],
) -> Dict[str, bool]:
    identity = dict(second_probe.get("unknown_game_identity", {}) or {})
    summary = dict(second_probe.get("summary", {}) or {})
    return {
        "source_sage5_loop_passed": bool(
            source_sage5.get("summary", {}).get("all_budgets_gate_passed", False)
        ),
        "source_a33_2_registration_complete": bool(
            source_a33_2.get("registration_performed", False)
        ),
        "second_game_differs_from_source": bool(transfer_guard.get("different_game")),
        "selected_game_unknown": bool(identity.get("unknown_game", False)),
        "selected_game_no_human_trace": bool(
            identity.get("no_human_trace_for_game", False)
        ),
        "selected_game_no_m2_trace": bool(
            identity.get("no_m2_arc_lewm_trace_for_game", False)
        ),
        "selected_game_no_specific_prior": bool(
            identity.get("no_game_specific_prior", False)
        ),
        "selected_game_matches_probe": str(identity.get("game_id", ""))
        == selected_game_id,
        "all_second_game_budgets_passed": bool(
            summary.get("all_budgets_gate_passed", False)
        ),
        "selected_actions_always_legal": all(
            bool(row.get("gate", {}).get("selected_action_always_legal", False))
            for row in second_probe.get("per_budget_results", []) or []
        ),
        "subgoal_switches_happened": all(
            bool(row.get("gate", {}).get("subgoal_switches_happened", False))
            for row in second_probe.get("per_budget_results", []) or []
        ),
        "no_catastrophic_repeat_collapse": all(
            bool(row.get("gate", {}).get("no_catastrophic_repeat_collapse", False))
            for row in second_probe.get("per_budget_results", []) or []
        ),
        "terminal_rate_under_threshold": all(
            bool(row.get("gate", {}).get("terminal_rate_under_threshold", False))
            for row in second_probe.get("per_budget_results", []) or []
        ),
        "scoped_registry_quarantined": bool(
            transfer_guard.get("quarantine_passed", False)
        )
        and int(transfer_guard.get("source_scoped_mechanics_reused", 0) or 0) == 0,
        "no_cross_game_mechanics_imported": int(
            transfer_guard.get("cross_game_mechanics_imported", 0) or 0
        )
        == 0,
    }


def summarize_sage6_transfer(
    *,
    source_sage5: Mapping[str, Any],
    source_a33_2: Mapping[str, Any],
    selection_audit: Sequence[Mapping[str, Any]],
    selected_game_id: str,
    second_probe: Mapping[str, Any],
    transfer_guard: Mapping[str, Any],
) -> Dict[str, Any]:
    per_budget = [
        row
        for row in second_probe.get("per_budget_results", []) or []
        if isinstance(row, Mapping)
    ]
    metrics = [dict(row.get("metrics", {}) or {}) for row in per_budget]
    terminal_rates = [float(row.get("terminal_rate", 0.0) or 0.0) for row in metrics]
    repeat_rates = [
        float(row.get("repeated_action_arg_rate", 0.0) or 0.0) for row in metrics
    ]
    unique_signatures = [
        int(row.get("unique_state_signatures", 0) or 0) for row in metrics
    ]
    return {
        "source_game_id": str(source_sage5.get("config", {}).get("game_id", "")),
        "selected_second_game_id": selected_game_id,
        "candidate_games_audited": len(selection_audit),
        "eligible_unknown_games": sum(
            str(row.get("eligibility_status", "")) == ELIGIBLE_UNKNOWN_GAME
            for row in selection_audit
        ),
        "known_or_seen_candidates_excluded": sum(
            str(row.get("eligibility_status", "")) == EXCLUDED_KNOWN_GAME
            for row in selection_audit
        ),
        "source_or_registry_scope_candidates_excluded": sum(
            str(row.get("eligibility_status", ""))
            in {EXCLUDED_SOURCE_GAME, EXCLUDED_SCOPE_LOCKED_REGISTRY_GAME}
            for row in selection_audit
        ),
        "selection_policy": DEFAULT_SELECTION_POLICY,
        "outcome_metrics_read_for_selection": False,
        "source_a33_2_entries_quarantined": int(
            source_a33_2.get("summary", {}).get(
                "scoped_confirmed_mechanics_registered", 0
            )
            or 0
        ),
        "source_scoped_mechanics_reused": int(
            transfer_guard.get("source_scoped_mechanics_reused", 0) or 0
        ),
        "cross_game_mechanics_imported": int(
            transfer_guard.get("cross_game_mechanics_imported", 0) or 0
        ),
        "budgets_evaluated": [int(row.get("budget", 0) or 0) for row in per_budget],
        "budgets_gate_passed": sum(
            bool(row.get("gate", {}).get("gate_passed", False)) for row in per_budget
        ),
        "budgets_total": len(per_budget),
        "all_budgets_gate_passed": bool(
            second_probe.get("summary", {}).get("all_budgets_gate_passed", False)
        ),
        "subgoal_switches_total": sum(
            int(row.get("subgoal_switches", 0) or 0) for row in metrics
        ),
        "budgets_with_progress_stall_detected": [
            int(per_budget[index].get("budget", 0) or 0)
            for index, row in enumerate(metrics)
            if bool(row.get("progress_stall_detected", False))
        ],
        "max_terminal_rate": max(terminal_rates, default=0.0),
        "max_repeated_action_arg_rate": max(repeat_rates, default=0.0),
        "min_unique_state_signatures": min(unique_signatures, default=0),
        "max_unique_state_signatures": max(unique_signatures, default=0),
        "levels_completed_max": max(
            (int(row.get("levels_completed", 0) or 0) for row in metrics),
            default=0,
        ),
        "support": 0,
        "truth_status": SAGE6_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
    }


def validate_sage5_transfer_source(source: Mapping[str, Any]) -> None:
    config = dict(source.get("config", {}) or {})
    identity = dict(source.get("unknown_game_identity", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != SAGE5_SCHEMA_VERSION:
        raise ValueError("SAGE.5 schema version is not supported by SAGE.6")
    if str(source.get("status", "")) != "UNRESOLVED":
        raise ValueError("SAGE.5 source must remain unresolved")
    if str(source.get("truth_status", "")) != SAGE5_TRUTH_STATUS:
        raise ValueError("SAGE.5 source truth must remain unevaluated")
    if str(source.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.5 source must remain candidate-only")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.5 source support must remain 0")
    if bool(source.get("revision_performed", False)):
        raise ValueError("SAGE.5 source cannot perform revision")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.5 source wrong_confirmations must remain 0")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.5 source cannot write A32/A33")
    if bool(source.get("policy_result_counted_as_confirmation", False)):
        raise ValueError("SAGE.5 policy result cannot count as confirmation")
    if str(source.get("outcome_status", "")) != SAGE5_ALL_BUDGETS_PASSED:
        raise ValueError("SAGE.6 requires a fully passed bounded source probe")
    if not bool(source.get("outcome_status_is_candidate_only", False)):
        raise ValueError("SAGE.5 outcome must remain explicitly candidate-only")
    if not bool(identity.get("unknown_game", False)) or not bool(
        identity.get("no_human_trace_for_game", False)
    ) or not bool(identity.get("no_m2_arc_lewm_trace_for_game", False)):
        raise ValueError("SAGE.5 source must pass unknown-game hygiene")
    game_id = str(config.get("game_id", ""))
    if not game_id or str(identity.get("game_id", "")) != game_id or str(
        summary.get("game_id", "")
    ) != game_id:
        raise ValueError("SAGE.5 source game identity must align")
    if not bool(summary.get("all_budgets_gate_passed", False)):
        raise ValueError("SAGE.5 source budgets must all pass")
    rows = [
        row
        for row in source.get("per_budget_results", []) or []
        if isinstance(row, Mapping)
    ]
    if not rows or len(rows) != int(summary.get("budgets_total", 0) or 0):
        raise ValueError("SAGE.5 source budget count must be exact")
    if any(not bool(row.get("gate", {}).get("gate_passed", False)) for row in rows):
        raise ValueError("every SAGE.5 source budget gate must pass")
    if any(int(row.get("metrics", {}).get("support", 0) or 0) != 0 for row in rows):
        raise ValueError("SAGE.5 per-budget support must remain 0")


def validate_a33_2_transfer_source(
    source: Mapping[str, Any],
    *,
    source_sage5: Mapping[str, Any],
) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != A33_2_SCHEMA_VERSION:
        raise ValueError("A33.2 schema version is not supported by SAGE.6")
    if str(source.get("outcome_status", "")) != A33_2_ENTRY_ADDED:
        raise ValueError("SAGE.6 requires the completed A33.2 registry")
    if str(source.get("status", "")) != "REGISTERED":
        raise ValueError("A33.2 source must be registered")
    if str(source.get("truth_status", "")) != A33_2_TRUTH_STATUS:
        raise ValueError("A33.2 source truth status must remain non-reevaluated")
    if not bool(source.get("registration_performed", False)):
        raise ValueError("A33.2 registration must be complete before SAGE.6")
    if bool(source.get("revision_performed", False)) or bool(
        source.get("confirmation_performed", False)
    ) or bool(source.get("refutation_performed", False)):
        raise ValueError("A33.2 cannot create a new scientific verdict")
    if not bool(source.get("a33_write_performed", False)):
        raise ValueError("A33.2 registry artifact must have been written")
    if bool(source.get("source_a32_5_mutated", False)) or bool(
        source.get("legacy_a33_1_registry_mutated", False)
    ):
        raise ValueError("A33.2 cannot mutate its sources or legacy registry")
    if bool(source.get("scope_generalization_performed", False)):
        raise ValueError("A33.2 cannot generalize its registered scope")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("A33.2 wrong_confirmations must remain 0")
    if str(source.get("support_origin", "")) != "A32.5_DECISION_RECORDS_ONLY":
        raise ValueError("A33.2 support must retain its A32.5 origin")
    entries = [
        row
        for row in source.get("scoped_confirmed_mechanics", []) or []
        if isinstance(row, Mapping)
    ]
    if not entries or len(entries) != int(
        summary.get("scoped_confirmed_mechanics_registered", 0) or 0
    ):
        raise ValueError("A33.2 scoped registry entry count must be exact")
    source_game_id = str(source_sage5.get("config", {}).get("game_id", ""))
    for entry in entries:
        if str(entry.get("status", "")) != "confirmed":
            raise ValueError("A33.2 transfer source accepts only confirmed entries")
        if str(entry.get("game_id", "")) != source_game_id:
            raise ValueError("A33.2 entries must remain scoped to the SAGE.5 game")
        if not all(
            bool(entry.get(flag, False))
            for flag in (
                "scope_game_locked",
                "scope_candidate_locked",
                "scope_contexts_locked",
                "scope_measurement_locked",
                "not_generalized_beyond_game",
                "not_generalized_beyond_candidate_scope",
                "not_generalized_to_other_actions",
            )
        ):
            raise ValueError("A33.2 entry scope must remain fully locked")
        if bool(entry.get("a33_confirmation_performed", False)):
            raise ValueError("A33.2 cannot reconfirm a scoped entry")
    excluded = [
        row
        for row in source.get("excluded_candidate_audit", []) or []
        if isinstance(row, Mapping)
    ]
    if not excluded or any(bool(row.get("registered", False)) for row in excluded):
        raise ValueError("A33.2 unresolved candidates must remain excluded")


def write_sage6_second_unknown_game_transfer(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _same_game(left: str, right: str) -> bool:
    return str(left).split("-", 1)[0] == str(right).split("-", 1)[0]


def _load_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _load_optional_json(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return _load_json(source)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SAGE.6 on a fixed-order second unknown game."
    )
    parser.add_argument("--source-sage5", default=str(DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH))
    parser.add_argument(
        "--source-a33-2",
        default=str(DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH),
    )
    parser.add_argument(
        "--m2-dataset-manifest",
        default=str(DEFAULT_M2_ARC_LEWM_DATASET_MANIFEST_PATH),
    )
    parser.add_argument("--human-traces-dir", default=str(DEFAULT_HUMAN_TRACES_DIR))
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH))
    parser.add_argument("--candidate-game-ids", nargs="+", default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_sage6_second_unknown_game_transfer(
        source_sage5_path=args.source_sage5,
        source_a33_2_path=args.source_a33_2,
        m2_dataset_manifest_path=args.m2_dataset_manifest,
        human_traces_dir=args.human_traces_dir,
        environments_dir=args.environments_dir,
        output_path=args.out,
        candidate_game_ids=args.candidate_game_ids,
        budgets=args.budgets,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
