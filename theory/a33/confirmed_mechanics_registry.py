"""A33 confirmed mechanics registry.

A33 turns A32 decisions into a compact registry of confirmed mechanics. It does
not read M3 artifacts directly; the registry is derived only from A32 decisions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.a32.revision_decisions import (
    DEFAULT_A32_REVISION_DECISIONS_OUTPUT_PATH,
    REVISION_ACCEPTED_AS_CONFIRMED,
)


DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH = (
    Path("diagnostics") / "a33" / "confirmed_mechanics_registry.json"
)


@dataclass(frozen=True)
class ConfirmedMechanicRegistryEntry:
    """One confirmed mechanic exported from A32 decisions."""

    key: str
    game_id: str
    action: str
    mechanic_family: str
    predicted_metric: str
    confirmed_support_independent: int
    experiments_spent: int
    control_actions_used: Tuple[str, ...]
    known_scope: str = "local_context"
    source_decision: str = REVISION_ACCEPTED_AS_CONFIRMED
    source_artifact: str = ""
    reused_control_support_excluded: int = 0
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False
    unresolved_candidates_excluded: bool = True
    support_reused_as_independent: bool = False
    status: str = "confirmed"
    evidence_notes: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "game_id": self.game_id,
            "action": self.action,
            "mechanic_family": self.mechanic_family,
            "predicted_metric": self.predicted_metric,
            "confirmed_support_independent": int(
                self.confirmed_support_independent
            ),
            "experiments_spent": int(self.experiments_spent),
            "control_actions_used": list(self.control_actions_used),
            "known_scope": self.known_scope,
            "source_decision": self.source_decision,
            "source_artifact": self.source_artifact,
            "reused_control_support_excluded": int(
                self.reused_control_support_excluded
            ),
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
            "unresolved_candidates_excluded": self.unresolved_candidates_excluded,
            "support_reused_as_independent": self.support_reused_as_independent,
            "status": self.status,
            "evidence_notes": list(self.evidence_notes),
        }


def run_confirmed_mechanics_registry_generation(
    *,
    decisions_path: str | Path = DEFAULT_A32_REVISION_DECISIONS_OUTPUT_PATH,
) -> Dict[str, Any]:
    """Build the confirmed-mechanics registry from A32 decisions."""
    payload = _load_json(decisions_path)
    entries = build_confirmed_mechanics_registry(
        payload,
        source_artifact=str(decisions_path),
    )
    return {
        "config": {
            "decisions_path": str(decisions_path),
        },
        "summary": summarize_confirmed_mechanics_registry(entries),
        "confirmed_mechanics": [entry.to_dict() for entry in entries],
        "source_stage": "A32",
        "registry_stage": "A33",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "unresolved_candidates_excluded": True,
    }


def build_confirmed_mechanics_registry(
    decisions_payload: Mapping[str, Any],
    *,
    source_artifact: str = "",
) -> Tuple[ConfirmedMechanicRegistryEntry, ...]:
    """Extract confirmed mechanics from A32 decision rows."""
    entries: list[ConfirmedMechanicRegistryEntry] = []
    for decision in decisions_payload.get("revision_decisions", []) or []:
        if not isinstance(decision, Mapping):
            continue
        if str(decision.get("decision", "")) != REVISION_ACCEPTED_AS_CONFIRMED:
            continue
        decision_record = dict(decision.get("decision_record", {}) or {})
        if str(decision_record.get("status", "")).lower() != "confirmed":
            continue
        spec = mechanic_prediction_spec(str(decision.get("key", "")))
        control_evidence = [
            dict(row)
            for row in decision.get("control_evidence", []) or []
            if isinstance(row, Mapping)
        ]
        control_actions = tuple(
            sorted(
                {
                    str(row.get("control_action", ""))
                    for row in control_evidence
                    if int(row.get("independent_support_events", 0) or 0) > 0
                    and str(row.get("control_action", ""))
                }
            )
        )
        evidence_summary = dict(decision.get("evidence_summary", {}) or {})
        entries.append(
            ConfirmedMechanicRegistryEntry(
                key=str(decision.get("key", "")),
                game_id=spec["game_id"],
                action=spec["action"],
                mechanic_family=spec["mechanic_family"],
                predicted_metric=spec["predicted_metric"],
                confirmed_support_independent=int(
                    decision_record.get("support", 0) or 0
                ),
                experiments_spent=int(
                    decision_record.get("experiments_spent", 0) or 0
                ),
                control_actions_used=control_actions,
                source_artifact=source_artifact,
                reused_control_support_excluded=int(
                    evidence_summary.get("reused_control_support_events", 0) or 0
                ),
                evidence_notes=(
                    "confirmed_support_uses_independent_controls_only",
                    "reused_control_support_excluded_from_independent_support",
                    "trace_and_prior_support_not_counted",
                ),
            )
        )
    return tuple(entries)


def mechanic_prediction_spec(key: str) -> Dict[str, str]:
    parts = key.split("::")
    if len(parts) >= 5 and parts[0] == "mechanic_prediction":
        return {
            "game_id": parts[1],
            "action": parts[2],
            "mechanic_family": parts[3],
            "predicted_metric": parts[4],
        }
    return {
        "game_id": "",
        "action": "",
        "mechanic_family": "",
        "predicted_metric": "",
    }


def summarize_confirmed_mechanics_registry(
    entries: Sequence[ConfirmedMechanicRegistryEntry],
) -> Dict[str, Any]:
    return {
        "confirmed_mechanics": len(entries),
        "confirmed_support_independent_total": sum(
            entry.confirmed_support_independent for entry in entries
        ),
        "experiments_spent_total": sum(entry.experiments_spent for entry in entries),
        "reused_control_support_excluded_total": sum(
            entry.reused_control_support_excluded for entry in entries
        ),
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "unresolved_candidates_excluded": True,
        "known_scope": "local_context",
    }


def write_confirmed_mechanics_registry(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build A33 confirmed-mechanics registry from A32 decisions.",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=DEFAULT_A32_REVISION_DECISIONS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_confirmed_mechanics_registry_generation(
        decisions_path=args.decisions,
    )
    write_confirmed_mechanics_registry(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
