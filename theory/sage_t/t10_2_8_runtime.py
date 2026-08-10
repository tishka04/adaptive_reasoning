"""Lineage-aware offline compile and pre-fit QA runtime for SAGE.T10.2.8."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _kernel_protocol
from . import t10_2_1_runtime as _kernel_runtime
from . import t10_2_7_protocol as _predecessor_protocol
from . import t10_2_7_runtime as _predecessor_runtime
from . import t10_2_8_protocol as _protocol

FORMAT_VERSION = "sage-t10.2.8-runtime-v1"
LINEAGE_AUDIT_FORMAT_VERSION = "sage-t10.2.8-lineage-audit-v1"
QA_REPORT_FORMAT_VERSION = "sage-t10.2.8-qa-report-v1"
TERMINAL_REPORT_FORMAT_VERSION = "sage-t10.2.8-terminal-report-v1"
LINEAGE_AUDIT_FILENAME = "lineage_audit.json"
QA_REPORT_FILENAME = "qa_report.json"
TERMINAL_REPORT_FILENAME = "t10_2_8_report.json"

canonical_json = _kernel_protocol.canonical_json
canonical_sha256 = _kernel_protocol.canonical_sha256
signed_payload = _kernel_protocol.signed_payload
ManifestDriftError = _kernel_protocol.ManifestDriftError
ProtocolError = _kernel_protocol.ProtocolError
JournalIntegrityError = _kernel_runtime.JournalIntegrityError
_science = _kernel_protocol._t10_2


def _write_once_payload(
    path: Path, payload: Mapping[str, Any], *, checksum_key: str
) -> None:
    if path.is_file():
        existing = _kernel_protocol._read_signed_json(path, checksum_key=checksum_key)
        if existing != dict(payload):
            raise JournalIntegrityError(f"immutable T10.2.8 artifact drifted: {path}")
        return
    _kernel_runtime._write_once(path, payload)


@contextmanager
def _recovery_seed_binding(seeds: Sequence[int]):
    """Extend only the frozen source split recognizer for lineage validation."""

    original = _science.CONFIRMATION_SEEDS
    additions = tuple(int(seed) for seed in seeds if int(seed) not in original)
    _science.CONFIRMATION_SEEDS = (*original, *additions)
    try:
        yield
    finally:
        _science.CONFIRMATION_SEEDS = original


def _load_execution_context(
    *, manifest_path: str | Path, repo_root: str | Path | None
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    root = _protocol._root(repo_root)
    manifest = _protocol.load_manifest(manifest_path, repo_root=root)
    predecessor = _predecessor_protocol.load_manifest(
        root / _predecessor_protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        repo_root=root,
        verify_repository=False,
        verify_live_migration=False,
    )
    if predecessor.get("manifest_checksum") != _protocol.PREDECESSOR_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.2.8 runtime predecessor drifted")
    kernel = _protocol._kernel_manifest(root)
    destination = root / _predecessor_protocol.DEFAULT_RECOVERY_ROOT
    return root, manifest, predecessor, kernel, destination


def _read_accepted_events(
    *, destination: Path, receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    path = destination / _predecessor_runtime.ACCEPTED_EVENT_FILENAME
    descriptor = _protocol._artifact_descriptor(path)
    if descriptor != receipt.get("accepted_event_ledger"):
        raise ManifestDriftError("T10.2.8 accepted ledger descriptor drifted")
    events = _science.read_event_ledger(path)
    if len(events) != int(
        receipt["predecessor_collection"]["accepted_event_count"]
    ):
        raise JournalIntegrityError("T10.2.8 accepted ledger count drifted")
    return events


def _lineage_key(lane: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(lane.get("split", "")),
        str(lane.get("game_id", "")),
        int(lane.get("seed", -1)),
    )


def build_lineage_audit(
    *,
    manifest: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    kernel: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = manifest["handoff_receipt"]
    registry = receipt["lineage_registry"]
    by_key = {_lineage_key(item["physical_lane"]): item for item in registry}
    counts: Counter[tuple[str, str, int]] = Counter()
    lineage_counts: Counter[str] = Counter()
    event_ids: list[str] = []
    parent_events: list[Mapping[str, Any]] = []
    recovery_events: list[Mapping[str, Any]] = []
    errors: list[str] = []
    environment = str(receipt["scientific_environment_sha256"])
    for row in events:
        try:
            key = (
                str(row.get("split", "")),
                str(row.get("game_id", "")),
                int(row.get("seed", -1)),
            )
            lineage = by_key.get(key)
            if lineage is None:
                errors.append(f"unregistered_physical_lane:{key}")
                continue
            provenance = row.get("provenance")
            if not isinstance(provenance, Mapping):
                errors.append(f"missing_provenance:{row.get('event_id', '')}")
                continue
            if provenance.get("manifest_checksum") != lineage[
                "provenance_manifest_checksum"
            ]:
                errors.append(f"manifest_lineage_mismatch:{row.get('event_id', '')}")
            if provenance.get("environment_sha256") != environment:
                errors.append(f"environment_lineage_mismatch:{row.get('event_id', '')}")
            counts[key] += 1
            lineage_name = str(lineage["lineage"])
            lineage_counts[lineage_name] += 1
            event_ids.append(str(row.get("event_id", "")))
            if lineage_name == "t10_2_7_recovery":
                recovery_events.append(row)
            else:
                parent_events.append(row)
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"malformed_event:{type(exc).__name__}")
    lane_counts_match = all(
        counts[_lineage_key(item["physical_lane"])]
        == int(item["expected_event_count"])
        for item in registry
    )
    schema_validation_passed = False
    schema_error: str | None = None
    if not errors and lane_counts_match:
        try:
            _science.validate_source_events(
                parent_events,
                manifest=kernel,
                replay=False,
            )
            recovery_seeds = [
                int(item["physical_lane"]["seed"])
                for item in registry
                if item["lineage"] == "t10_2_7_recovery"
            ]
            execution_manifest = _predecessor_runtime.build_execution_manifest(
                protocol_manifest=predecessor,
                kernel=kernel,
            )
            with _recovery_seed_binding(recovery_seeds):
                _science.validate_source_events(
                    recovery_events,
                    manifest=execution_manifest,
                    replay=False,
                )
            schema_validation_passed = True
        except (ProtocolError, OSError, ValueError, KeyError) as exc:
            schema_error = f"{type(exc).__name__}:{exc}"
    checks = {
        "handoff_event_count": len(events)
        == int(receipt["predecessor_collection"]["accepted_event_count"]),
        "lineage_registry_bound": receipt["lineage_registry_sha256"]
        == canonical_sha256(registry),
        "all_events_registered": not errors,
        "per_lane_event_counts": lane_counts_match,
        "event_ids_present_and_unique": bool(event_ids)
        and all(event_ids)
        and len(event_ids) == len(set(event_ids)),
        "parent_lineage_present": lineage_counts["t10_2_2_parent"] > 0,
        "recovery_lineage_present": lineage_counts["t10_2_7_recovery"] > 0,
        "event_schema_and_provenance": schema_validation_passed,
        "physical_replay_absent": receipt["predecessor_collection"][
            "replayed_physical_actions"
        ]
        == 0,
        "firewall_closed": True,
    }
    return signed_payload(
        {
            "format_version": LINEAGE_AUDIT_FORMAT_VERSION,
            "phase": "lineage_audit",
            "status": (
                "PASS_T10_2_8_LINEAGE"
                if all(checks.values())
                else "DATA_OR_PROVENANCE_INVALID"
            ),
            "manifest_checksum": manifest["manifest_checksum"],
            "handoff_receipt_checksum": receipt["receipt_checksum"],
            "event_count": len(events),
            "event_ids_sha256": canonical_sha256(sorted(event_ids)),
            "lineage_event_counts": dict(sorted(lineage_counts.items())),
            "registered_lane_count": len(registry),
            "validation_errors": errors[:32],
            "schema_error": schema_error,
            "checks": checks,
            "passed": all(checks.values()),
            "firewall": {
                "environment_calls": 0,
                "physical_actions": 0,
                "physical_replay": 0,
                "model_fit_opened": False,
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
            },
        },
        checksum_key="audit_checksum",
    )


def _behavior_diagnostics(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_controller: dict[str, Counter[str]] = defaultdict(Counter)
    progression_values: list[float] = []
    for row in events:
        selection = row.get("selection", {})
        outcome = row.get("outcome", {})
        labels = row.get("labels", {})
        controller = (
            str(selection.get("controller", "unknown"))
            if isinstance(selection, Mapping)
            else "unknown"
        )
        outcome = outcome if isinstance(outcome, Mapping) else {}
        labels = labels if isinstance(labels, Mapping) else {}
        progression = float(outcome.get("progression", 0.0))
        progression_values.append(progression)
        counts = by_controller[controller]
        counts["events"] += 1
        counts["positive_progress"] += progression > 0.0
        counts["goals"] += outcome.get("goal") is True
        counts["terminals"] += outcome.get("terminal") is True
        counts["level_complete_labels"] += labels.get("level_complete") is True
        if isinstance(selection, Mapping):
            counts["decision_engine_used"] += selection.get("decision_engine_used") is True
            counts["option_conditioned"] += selection.get("option_conditioned") is True
    positive = sum(value > 0.0 for value in progression_values)
    goals = sum(
        isinstance(row.get("outcome"), Mapping)
        and row["outcome"].get("goal") is True
        for row in events
    )
    return {
        "diagnostic_only": True,
        "event_count": len(events),
        "positive_progress_event_count": positive,
        "goal_event_count": goals,
        "maximum_progression": max(progression_values, default=0.0),
        "diagnosis": (
            "ZERO_OBSERVED_PROGRESS"
            if positive == 0 and goals == 0
            else "OBSERVED_PROGRESS_PRESENT"
        ),
        "by_controller": {
            key: dict(sorted(value.items())) for key, value in sorted(by_controller.items())
        },
    }


def build_qa_report(
    *, manifest: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compute the frozen QA metrics after a separate lineage validation."""

    gate = manifest["qa_gate"]
    correspondence_denominator = sum(
        int(_science._mapping(row.get("correspondence")).get("fraction_denominator", 0))
        for row in events
    )
    confident_matches = sum(
        int(_science._mapping(row.get("correspondence")).get("confident_matches", 0))
        for row in events
    )
    fully_ambiguous_matches = sum(
        int(
            _science._mapping(row.get("correspondence")).get(
                "fully_ambiguous_matches", 0
            )
        )
        for row in events
    )
    confident_fraction = (
        confident_matches / correspondence_denominator
        if correspondence_denominator
        else 0.0
    )
    ambiguous_fraction = (
        fully_ambiguous_matches / correspondence_denominator
        if correspondence_denominator
        else 1.0
    )

    exact_transport_certificates: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    noncomparable_transport_certificates = 0
    certificate_schema_valid = True
    for row in events:
        certificates = row.get("transport_certificates")
        if not isinstance(certificates, (list, tuple)):
            certificate_schema_valid = False
            continue
        for certificate in certificates:
            if not isinstance(certificate, Mapping) or certificate.get("exact") not in (
                True,
                False,
            ):
                certificate_schema_valid = False
                continue
            if certificate.get("exact") is True:
                if (
                    certificate.get("live_graph_exact_attested") is not True
                    or certificate.get("summary_commutative_exact") is not True
                    or certificate.get("certifies_gauge_equivalence") is not True
                ):
                    certificate_schema_valid = False
                    continue
                exact_transport_certificates.append((row, certificate))
            else:
                if certificate.get("comparable") is not False:
                    certificate_schema_valid = False
                noncomparable_transport_certificates += 1
    transport_round_trip = all(
        certificate.get("round_trip_exact") is True
        and certificate.get("certifies_gauge_equivalence") is True
        for _row, certificate in exact_transport_certificates
    )
    permutation_invariant = all(
        _science._boolean(row, "transport", "entity_permutation_invariant", default=False)
        for row, _certificate in exact_transport_certificates
    )
    commutative = all(
        isinstance(certificate.get("commutativity"), Mapping)
        and certificate["commutativity"].get("exact") is True
        and certificate.get("summary_commutative_exact") is True
        and _science._boolean(
            row, "transport", "summary_commutative_exact", default=False
        )
        for row, certificate in exact_transport_certificates
    )

    predicate_counts: Counter[str] = Counter()
    predicate_totals: Counter[str] = Counter()
    predicate_games: dict[str, set[str]] = defaultdict(set)
    declarations = [_science._declared_learned_predicates(row) for row in events]
    declared_predicates = (
        frozenset().union(*declarations) if declarations else frozenset()
    )
    declaration_consistent = bool(declarations) and all(
        declaration == declared_predicates and bool(declaration)
        for declaration in declarations
    )
    label_coverage_complete = True
    for row in events:
        labels = row.get("labels", {})
        if not isinstance(labels, Mapping):
            label_coverage_complete = False
            continue
        normalized_labels = {
            str(name).casefold(): value for name, value in labels.items()
        }
        for name in declared_predicates:
            value = normalized_labels.get(name)
            if value not in (True, False, 0, 1):
                label_coverage_complete = False
                continue
            predicate_totals[name] += 1
            if bool(value):
                predicate_counts[name] += 1
                predicate_games[name].add(str(row["game_id"]))

    predicate_metrics: dict[str, dict[str, Any]] = {}
    predicate_checks: dict[str, bool] = {}
    for name in sorted(predicate_totals):
        total = predicate_totals[name]
        positives = predicate_counts[name]
        prevalence = positives / total if total else 0.0
        games = len(predicate_games[name])
        predicate_metrics[name] = {
            "games": games,
            "positives": positives,
            "prevalence": prevalence,
            "total": total,
        }
        predicate_checks[name] = bool(
            float(gate["minimum_predicate_prevalence"])
            <= prevalence
            <= float(gate["maximum_predicate_prevalence"])
            and positives >= int(gate["minimum_predicate_support"])
            and games >= int(gate["minimum_predicate_games"])
        )

    nonterminal = 0
    evaluable = 0
    coherent = 0
    for row in events:
        prefix = row.get("prefix", {})
        if not isinstance(prefix, Mapping):
            prefix = {}
        is_nonterminal = bool(
            prefix.get("nonterminal", row.get("nonterminal_prefix", False))
        )
        if not is_nonterminal:
            continue
        nonterminal += 1
        if bool(prefix.get("evaluable", row.get("evaluable_nonterminal", False))):
            evaluable += 1
        coherent_frames = sum(
            projection.get("complete") is True
            for projection in _science._mapping(row.get("projections")).values()
            if isinstance(projection, Mapping)
        )
        if coherent_frames >= 2:
            coherent += 1
    evaluable_fraction = evaluable / nonterminal if nonterminal else 0.0
    coherent_fraction = coherent / nonterminal if nonterminal else 0.0

    checks = {
        "events_present": bool(events),
        "persistent_correspondence": confident_fraction
        >= float(gate["minimum_confident_correspondence"]),
        "fully_ambiguous_correspondence": ambiguous_fraction
        < float(gate["maximum_fully_ambiguous_correspondence"]),
        "transport_round_trip_exact": transport_round_trip,
        "entity_permutation_invariant": permutation_invariant,
        "transport_commutative_exact": commutative,
        "transport_certificate_schema": certificate_schema_valid,
        "exact_transport_evidence_present": bool(exact_transport_certificates),
        "learned_predicates_present": bool(declared_predicates),
        "learned_predicate_declaration_consistent": declaration_consistent,
        "learned_predicate_label_coverage": label_coverage_complete,
        "learned_predicate_prevalence_and_support": bool(predicate_checks)
        and all(predicate_checks.values()),
        "evaluable_nonterminal_prefixes": evaluable_fraction
        >= float(gate["minimum_evaluable_nonterminal_prefix_fraction"]),
        "multiframe_coherent_prefixes": coherent_fraction
        >= float(gate["minimum_multiframe_coherent_prefix_fraction"]),
        "holdout_closed": True,
        "ar25_closed": True,
        "source_validation_closed": True,
    }
    passed = all(checks.values())
    return signed_payload(
        {
            "format_version": QA_REPORT_FORMAT_VERSION,
            "phase": "offline_qa",
            "status": "PASS_T10_2_8_QA" if passed else "FAIL_T10_2_8_QA",
            "manifest_checksum": manifest["manifest_checksum"],
            "handoff_receipt_checksum": manifest["handoff_receipt"][
                "receipt_checksum"
            ],
            "event_count": len(events),
            "event_ids_sha256": canonical_sha256(
                sorted(_science._event_id(row) for row in events)
            ),
            "metrics": {
                "confident_correspondence_fraction": confident_fraction,
                "fully_ambiguous_correspondence_fraction": ambiguous_fraction,
                "correspondence_trials": correspondence_denominator,
                "confident_correspondence_matches": confident_matches,
                "fully_ambiguous_correspondence_matches": fully_ambiguous_matches,
                "nonterminal_prefixes": nonterminal,
                "evaluable_nonterminal_prefix_fraction": evaluable_fraction,
                "multiframe_coherent_prefix_fraction": coherent_fraction,
                "predicates": predicate_metrics,
                "declared_learned_predicates": sorted(declared_predicates),
                "exact_or_invertible_transports": len(exact_transport_certificates),
                "partial_noncomparable_transports": (
                    noncomparable_transport_certificates
                ),
            },
            "behavior_diagnostics": _behavior_diagnostics(events),
            "checks": checks,
            "failed_checks": [key for key, value in checks.items() if not value],
            "passed": passed,
            "fit_authorized": False,
            "firewall": {
                "environment_calls": 0,
                "physical_actions": 0,
                "physical_replay": 0,
                "model_fit_opened": False,
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
            },
        },
        checksum_key="report_checksum",
    )


def _not_evaluated_qa(
    *, manifest: Mapping[str, Any], lineage_audit: Mapping[str, Any]
) -> dict[str, Any]:
    return signed_payload(
        {
            "format_version": QA_REPORT_FORMAT_VERSION,
            "phase": "offline_qa",
            "status": "NOT_EVALUATED_LINEAGE_FAILURE",
            "manifest_checksum": manifest["manifest_checksum"],
            "handoff_receipt_checksum": manifest["handoff_receipt"][
                "receipt_checksum"
            ],
            "lineage_audit_checksum": lineage_audit["audit_checksum"],
            "event_count": 0,
            "metrics": {},
            "behavior_diagnostics": {},
            "checks": {"lineage_validated_before_qa": False},
            "failed_checks": ["lineage_validated_before_qa"],
            "passed": False,
            "fit_authorized": False,
            "firewall": {
                "environment_calls": 0,
                "physical_actions": 0,
                "physical_replay": 0,
                "model_fit_opened": False,
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
            },
        },
        checksum_key="report_checksum",
    )


def compile_phase(
    *,
    manifest_path: str | Path = _protocol.DEFAULT_MANIFEST_PATH,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root, manifest, predecessor, kernel, predecessor_root = _load_execution_context(
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    destination = root / _protocol.DEFAULT_OUTPUT_ROOT
    terminal_path = destination / TERMINAL_REPORT_FILENAME
    if terminal_path.is_file():
        return _kernel_protocol._read_signed_json(
            terminal_path, checksum_key="terminal_checksum"
        )
    events = _read_accepted_events(
        destination=predecessor_root,
        receipt=manifest["handoff_receipt"],
    )
    lineage = build_lineage_audit(
        manifest=manifest,
        predecessor=predecessor,
        kernel=kernel,
        events=events,
    )
    qa = (
        build_qa_report(manifest=manifest, events=events)
        if lineage["passed"] is True
        else _not_evaluated_qa(manifest=manifest, lineage_audit=lineage)
    )
    destination.mkdir(parents=True, exist_ok=True)
    lineage_path = destination / LINEAGE_AUDIT_FILENAME
    qa_path = destination / QA_REPORT_FILENAME
    _write_once_payload(lineage_path, lineage, checksum_key="audit_checksum")
    _write_once_payload(qa_path, qa, checksum_key="report_checksum")
    lineage_passed = lineage["passed"] is True
    qa_passed = qa["passed"] is True
    terminal = signed_payload(
        {
            "format_version": TERMINAL_REPORT_FORMAT_VERSION,
            "phase": "compile",
            "status": (
                "PASS_T10_2_8_QA_READY_FOR_SEPARATE_SOURCE_TRAIN_PROTOCOL"
                if lineage_passed and qa_passed
                else "FAIL_T10_2_8_QA_STOP_BEFORE_FIT"
                if lineage_passed
                else "DATA_OR_PROVENANCE_INVALID"
            ),
            "manifest_checksum": manifest["manifest_checksum"],
            "handoff_receipt_checksum": manifest["handoff_receipt"][
                "receipt_checksum"
            ],
            "predecessor_collection_report_checksum": manifest[
                "handoff_receipt"
            ]["predecessor_collection"]["report_checksum"],
            "lineage_audit": _protocol._artifact_descriptor(lineage_path),
            "lineage_audit_checksum": lineage["audit_checksum"],
            "qa_report": _protocol._artifact_descriptor(qa_path),
            "qa_report_checksum": qa["report_checksum"],
            "lineage_passed": lineage_passed,
            "qa_passed": qa_passed,
            "passed": lineage_passed and qa_passed,
            "failed_qa_checks": list(qa.get("failed_checks", ())),
            "fit_authorized": False,
            "source_train_authorized": False,
            "next_protocol_authorized": lineage_passed and qa_passed,
            "stop_before_fit": not (lineage_passed and qa_passed),
            "physical_actions_executed": 0,
            "physical_actions_replayed": 0,
            "firewall": {
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
                "production_authority": False,
            },
        },
        checksum_key="terminal_checksum",
    )
    _write_once_payload(terminal_path, terminal, checksum_key="terminal_checksum")
    return terminal


def status_phase(
    *,
    manifest_path: str | Path = _protocol.DEFAULT_MANIFEST_PATH,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root, manifest, *_ = _load_execution_context(
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    destination = root / _protocol.DEFAULT_OUTPUT_ROOT
    terminal_path = destination / TERMINAL_REPORT_FILENAME
    if terminal_path.is_file():
        terminal = _kernel_protocol._read_signed_json(
            terminal_path, checksum_key="terminal_checksum"
        )
        return {
            "status": "COMPLETE_T10_2_8_OFFLINE_QA",
            "manifest_checksum": manifest["manifest_checksum"],
            "terminal_status": terminal["status"],
            "terminal_checksum": terminal["terminal_checksum"],
            "lineage_passed": terminal["lineage_passed"],
            "qa_passed": terminal["qa_passed"],
            "fit_authorized": False,
            "source_validation_opened": False,
            "ar25_opened": False,
        }
    return {
        "status": "READY_T10_2_8_OFFLINE_QA",
        "manifest_checksum": manifest["manifest_checksum"],
        "handoff": _protocol.verify_handoff_receipt_live(
            manifest["handoff_receipt"], repo_root=root
        ),
        "physical_actions_authorized": 0,
        "fit_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("status", "compile"))
    parser.add_argument("--manifest", default=str(_protocol.DEFAULT_MANIFEST_PATH))
    parser.add_argument("--repo-root", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = (
            status_phase(manifest_path=args.manifest, repo_root=args.repo_root)
            if args.phase == "status"
            else compile_phase(manifest_path=args.manifest, repo_root=args.repo_root)
        )
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(canonical_json({"error": f"{type(exc).__name__}:{exc}"}))
        return 2
    print(canonical_json(payload))
    if args.phase == "compile" and payload.get("status") != (
        "PASS_T10_2_8_QA_READY_FOR_SEPARATE_SOURCE_TRAIN_PROTOCOL"
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
