"""M1.3k controlled scientific integration pretest.

This module checks that M1.3j mechanic revision candidates can enter the
A15-A31 epistemic ledger without becoming confirmations or gaining artificial
support. It does not run a new experiment and does not revise beliefs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.epistemic_metrics import (
    HypothesisRecord,
    HypothesisStatus,
    MechanicsOracle,
    score_beliefs,
)

from .mechanic_revision_candidate import (
    DEFAULT_MECHANIC_REVISION_CANDIDATES_OUTPUT_PATH,
)

DEFAULT_SCIENTIFIC_INTEGRATION_PRETEST_OUTPUT_PATH = (
    Path("diagnostics") / "m1" / "scientific_integration_pretest.json"
)


@dataclass(frozen=True)
class ScientificLedgerEntry:
    """One M1 mechanic proposal admitted into the A15-A31 ledger."""

    revision_candidate_id: str
    game_id: str
    key: str
    description: str
    status: HypothesisStatus
    support: int = 0
    contradictions: int = 0
    experiments_spent: int = 0
    controlled_test_required: bool = True
    entered_scientific_ledger: bool = True
    observation_counted_as_confirmation: bool = False
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    def to_record(self) -> HypothesisRecord:
        return HypothesisRecord(
            key=self.key,
            description=self.description,
            status=self.status,
            support=int(self.support),
            contradictions=int(self.contradictions),
            experiments_spent=int(self.experiments_spent),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision_candidate_id": self.revision_candidate_id,
            "game_id": self.game_id,
            "key": self.key,
            "description": self.description,
            "status": self.status.value,
            "support": int(self.support),
            "contradictions": int(self.contradictions),
            "experiments_spent": int(self.experiments_spent),
            "controlled_test_required": self.controlled_test_required,
            "entered_scientific_ledger": self.entered_scientific_ledger,
            "observation_counted_as_confirmation": (
                self.observation_counted_as_confirmation
            ),
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


def run_scientific_integration_pretest(
    *,
    revision_candidates_path: str | Path = DEFAULT_MECHANIC_REVISION_CANDIDATES_OUTPUT_PATH,
) -> Dict[str, Any]:
    """Pretest M1 mechanic revision candidates in the A15-A31 ledger."""
    payload = json.loads(Path(revision_candidates_path).read_text(encoding="utf-8"))
    revisions = [dict(item) for item in payload.get("revision_candidates", []) or []]
    entries = [
        ledger_entry_from_revision_candidate(revision)
        for revision in revisions
        if bool(revision.get("a15_a31_ready"))
    ]
    score_by_game = score_entries_by_game(entries)
    return {
        "config": {
            "revision_candidates_path": str(revision_candidates_path),
        },
        "summary": summarize_scientific_integration(entries, score_by_game),
        "ledger_entries": [entry.to_dict() for entry in entries],
        "scores_by_game": {
            game_id: score.to_dict() for game_id, score in sorted(score_by_game.items())
        },
        "status": "UNRESOLVED",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
    }


def ledger_entry_from_revision_candidate(
    revision: Mapping[str, Any],
) -> ScientificLedgerEntry:
    proposal = dict(revision.get("a15_a31_revision_proposal", {}) or {})
    prediction = dict(revision.get("prediction", {}) or {})
    return ScientificLedgerEntry(
        revision_candidate_id=str(revision.get("revision_candidate_id", "")),
        game_id=str(prediction.get("game_id", "")),
        key=str(proposal.get("key", "")),
        description=str(proposal.get("description", "")),
        status=_normalize_status(proposal.get("proposed_status", "UNRESOLVED")),
        support=int(proposal.get("support", 0) or 0),
        contradictions=int(proposal.get("contradictions", 0) or 0),
        experiments_spent=int(proposal.get("experiments_spent", 0) or 0),
        controlled_test_required=bool(
            proposal.get("controlled_test_required", True)
        ),
        observation_counted_as_confirmation=bool(
            proposal.get("observation_counted_as_confirmation", False)
        ),
        trace_support_counted_as_proof=bool(
            revision.get("trace_support_counted_as_proof", False)
        ),
        prior_counted_as_proof=bool(revision.get("prior_counted_as_proof", False)),
    )


def score_entries_by_game(
    entries: Sequence[ScientificLedgerEntry],
) -> Dict[str, Any]:
    records_by_game: Dict[str, list[HypothesisRecord]] = {}
    for entry in entries:
        records_by_game.setdefault(entry.game_id, []).append(entry.to_record())
    return {
        game_id: score_beliefs(
            records,
            MechanicsOracle(game_id),
            experiment_actions=sum(record.experiments_spent for record in records),
        )
        for game_id, records in records_by_game.items()
    }


def summarize_scientific_integration(
    entries: Sequence[ScientificLedgerEntry],
    score_by_game: Mapping[str, Any],
) -> Dict[str, Any]:
    records = [entry.to_record() for entry in entries]
    return {
        "revision_candidates_total": len(entries),
        "entered_scientific_ledger": len(
            [entry for entry in entries if entry.entered_scientific_ledger]
        ),
        "unresolved_records": len(
            [record for record in records if record.status == HypothesisStatus("unresolved")]
        ),
        "confirmed_records": len(
            [record for record in records if record.status == HypothesisStatus("confirmed")]
        ),
        "refuted_records": len(
            [record for record in records if record.status == HypothesisStatus("refuted")]
        ),
        "support_total": sum(record.support for record in records),
        "contradictions_total": sum(record.contradictions for record in records),
        "experiments_spent_total": sum(record.experiments_spent for record in records),
        "controlled_tests_required": len(
            [entry for entry in entries if entry.controlled_test_required]
        ),
        "observation_counted_as_confirmation": any(
            entry.observation_counted_as_confirmation for entry in entries
        ),
        "trace_support_counted_as_proof": any(
            entry.trace_support_counted_as_proof for entry in entries
        ),
        "prior_counted_as_proof": any(entry.prior_counted_as_proof for entry in entries),
        "revision_performed": False,
        "wrong_confirmations": sum(
            int(score.wrong_confirmations) for score in score_by_game.values()
        ),
    }


def write_scientific_integration_pretest(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SCIENTIFIC_INTEGRATION_PRETEST_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalize_status(value: Any) -> HypothesisStatus:
    raw = str(value or "UNRESOLVED").strip().lower()
    if raw == "confirmed":
        return HypothesisStatus("confirmed")
    if raw == "refuted":
        return HypothesisStatus("refuted")
    return HypothesisStatus("unresolved")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M1.3k controlled scientific integration pretest.",
    )
    parser.add_argument(
        "--revision-candidates",
        type=Path,
        default=DEFAULT_MECHANIC_REVISION_CANDIDATES_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SCIENTIFIC_INTEGRATION_PRETEST_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_scientific_integration_pretest(
        revision_candidates_path=args.revision_candidates,
    )
    write_scientific_integration_pretest(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "trace_support_counted_as_proof": False,
                "prior_counted_as_proof": False,
                "observation_counted_as_confirmation": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
