"""M3 scientific planner state.

M3 reads M1 ledger entries and controlled-experiment artifacts, then builds a
small planning state. It aggregates evidence events, but never turns those
events into ledger support or a verdict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.m1.controlled_followup_experiment import (
    DEFAULT_CONTROLLED_EXPERIMENT_RESULTS_OUTPUT_PATH,
)
from theory.m1.scientific_integration_pretest import (
    DEFAULT_SCIENTIFIC_INTEGRATION_PRETEST_OUTPUT_PATH,
)


@dataclass(frozen=True)
class ScientificPlanningState:
    """Planning view over unresolved ledger entries and experiment events."""

    ledger_entries: Tuple[Dict[str, Any], ...]
    controlled_experiment_results: Tuple[Dict[str, Any], ...]
    remaining_budget: int
    tested_hypothesis_keys: Tuple[str, ...]
    support_events_by_key: Dict[str, int] = field(default_factory=dict)
    independent_support_events_by_key: Dict[str, int] = field(default_factory=dict)
    reused_control_support_events_by_key: Dict[str, int] = field(default_factory=dict)
    contradiction_events_by_key: Dict[str, int] = field(default_factory=dict)
    controlled_experiments_by_key: Dict[str, int] = field(default_factory=dict)
    controls_used_by_key: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    support_controls_by_key: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    skipped_controls: Tuple[Dict[str, Any], ...] = ()
    open_questions: Tuple[str, ...] = ()

    def summary(self) -> Dict[str, Any]:
        open_entries = [
            entry
            for entry in self.ledger_entries
            if str(entry.get("status", "")).lower() == "unresolved"
        ]
        open_keys = {str(entry.get("key", "")) for entry in open_entries}
        tested = set(self.tested_hypothesis_keys)
        return {
            "open_hypotheses": len(open_entries),
            "tested_hypotheses": len(open_keys & tested),
            "untested_hypotheses": len(open_keys - tested),
            "support_events_total": sum(self.support_events_by_key.values()),
            "independent_support_events_total": sum(
                self.independent_support_events_by_key.values()
            ),
            "reused_control_support_events_total": sum(
                self.reused_control_support_events_by_key.values()
            ),
            "contradiction_events_total": sum(
                self.contradiction_events_by_key.values()
            ),
            "controlled_experiments_run": sum(
                self.controlled_experiments_by_key.values()
            ),
            "remaining_budget": int(self.remaining_budget),
            "wrong_confirmations": 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ledger_entries": [dict(entry) for entry in self.ledger_entries],
            "controlled_experiment_results": [
                dict(experiment) for experiment in self.controlled_experiment_results
            ],
            "remaining_budget": int(self.remaining_budget),
            "tested_hypothesis_keys": list(self.tested_hypothesis_keys),
            "support_events_by_key": dict(self.support_events_by_key),
            "independent_support_events_by_key": dict(
                self.independent_support_events_by_key
            ),
            "reused_control_support_events_by_key": dict(
                self.reused_control_support_events_by_key
            ),
            "contradiction_events_by_key": dict(self.contradiction_events_by_key),
            "controlled_experiments_by_key": dict(self.controlled_experiments_by_key),
            "controls_used_by_key": {
                key: list(value) for key, value in self.controls_used_by_key.items()
            },
            "support_controls_by_key": {
                key: list(value) for key, value in self.support_controls_by_key.items()
            },
            "skipped_controls": [dict(item) for item in self.skipped_controls],
            "open_questions": list(self.open_questions),
            "summary": self.summary(),
        }


def build_scientific_planning_state(
    *,
    scientific_integration_path: str | Path = DEFAULT_SCIENTIFIC_INTEGRATION_PRETEST_OUTPUT_PATH,
    controlled_results_paths: Sequence[str | Path] = (
        DEFAULT_CONTROLLED_EXPERIMENT_RESULTS_OUTPUT_PATH,
    ),
    budget: int = 3,
    game_id: str | None = None,
) -> ScientificPlanningState:
    """Load M1/M3 artifacts and build the M3 planning state."""
    ledger_payload = _load_json(scientific_integration_path)
    controlled_payloads = [
        _load_json(path) for path in controlled_results_paths if Path(path).exists()
    ]
    return build_scientific_planning_state_from_payloads(
        ledger_payload=ledger_payload,
        controlled_payloads=controlled_payloads,
        budget=budget,
        game_id=game_id,
    )


def build_scientific_planning_state_from_payloads(
    *,
    ledger_payload: Mapping[str, Any],
    controlled_payloads: Sequence[Mapping[str, Any]] = (),
    budget: int = 3,
    game_id: str | None = None,
    extra_controlled_experiments: Sequence[Mapping[str, Any]] = (),
    skipped_controls: Sequence[Mapping[str, Any]] = (),
    open_questions: Sequence[str] = (),
) -> ScientificPlanningState:
    """Build state from already-loaded payloads.

    When a controlled payload contains detailed ``controlled_experiments``, those
    rows are the single source of evidence. ``updated_ledger_entries`` are read
    only as a fallback for older or reduced artifacts.
    """
    ledger_entries = _ledger_entries(ledger_payload, game_id=game_id)
    experiments: list[Dict[str, Any]] = []
    payload_skipped: list[Dict[str, Any]] = []
    payload_questions: list[str] = []

    for payload in controlled_payloads:
        experiments.extend(_experiments_from_controlled_payload(payload, game_id=game_id))
        payload_skipped.extend(_dict_rows(payload.get("skipped_controls", []) or []))
        payload_questions.extend(str(item) for item in payload.get("open_questions", []) or [])

    experiments.extend(_dict_rows(extra_controlled_experiments))
    payload_skipped.extend(_dict_rows(skipped_controls))
    payload_questions.extend(str(item) for item in open_questions)

    aggregates = _aggregate_experiments(experiments)
    spent = sum(aggregates["controlled_experiments_by_key"].values())
    return ScientificPlanningState(
        ledger_entries=tuple(ledger_entries),
        controlled_experiment_results=tuple(experiments),
        remaining_budget=max(0, int(budget) - int(spent)),
        tested_hypothesis_keys=tuple(sorted(aggregates["tested_hypothesis_keys"])),
        support_events_by_key=aggregates["support_events_by_key"],
        independent_support_events_by_key=aggregates[
            "independent_support_events_by_key"
        ],
        reused_control_support_events_by_key=aggregates[
            "reused_control_support_events_by_key"
        ],
        contradiction_events_by_key=aggregates["contradiction_events_by_key"],
        controlled_experiments_by_key=aggregates["controlled_experiments_by_key"],
        controls_used_by_key=aggregates["controls_used_by_key"],
        support_controls_by_key=aggregates["support_controls_by_key"],
        skipped_controls=tuple(_dedupe_dict_rows(payload_skipped)),
        open_questions=tuple(_dedupe_strings(payload_questions)),
    )


def updated_ledger_entries_from_state(
    state: ScientificPlanningState,
) -> Tuple[Dict[str, Any], ...]:
    """Return candidate-only ledger rows with event counts but support kept at 0."""
    rows: list[Dict[str, Any]] = []
    for entry in state.ledger_entries:
        key = str(entry.get("key", ""))
        rows.append(
            {
                "key": key,
                "game_id": str(entry.get("game_id", "")),
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "contradictions": 0,
                "support_events": int(state.support_events_by_key.get(key, 0)),
                "independent_support_events": int(
                    state.independent_support_events_by_key.get(key, 0)
                ),
                "reused_control_support_events": int(
                    state.reused_control_support_events_by_key.get(key, 0)
                ),
                "contradiction_events": int(
                    state.contradiction_events_by_key.get(key, 0)
                ),
                "controlled_experiments_run": int(
                    state.controlled_experiments_by_key.get(key, 0)
                ),
                "controlled_test_required": True,
                "revision_performed": False,
                "observation_counted_as_confirmation": False,
                "trace_support_counted_as_proof": False,
                "prior_counted_as_proof": False,
            }
        )
    return tuple(rows)


def _aggregate_experiments(
    experiments: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    support_events_by_key: Dict[str, int] = {}
    independent_support_events_by_key: Dict[str, int] = {}
    reused_control_support_events_by_key: Dict[str, int] = {}
    contradiction_events_by_key: Dict[str, int] = {}
    controlled_experiments_by_key: Dict[str, int] = {}
    controls_used: Dict[str, list[str]] = {}
    support_controls: Dict[str, list[str]] = {}
    tested: set[str] = set()

    for experiment in experiments:
        key = _experiment_key(experiment)
        if not key:
            continue
        support = int(experiment.get("support_events", 0) or 0)
        contradiction = int(experiment.get("contradiction_events", 0) or 0)
        run_count = int(experiment.get("controlled_experiments_run", 0) or 0)
        if run_count <= 0 and (support or contradiction or experiment.get("control_action")):
            run_count = 1

        control = str(experiment.get("control_action", "") or "")
        if run_count > 0:
            tested.add(key)
            controlled_experiments_by_key[key] = (
                controlled_experiments_by_key.get(key, 0) + run_count
            )
            if control:
                controls_used.setdefault(key, []).append(control)

        support_events_by_key[key] = support_events_by_key.get(key, 0) + support
        contradiction_events_by_key[key] = (
            contradiction_events_by_key.get(key, 0) + contradiction
        )

        if support <= 0:
            continue
        already_independent = set(support_controls.setdefault(key, []))
        reuse_reason = str(experiment.get("control_reuse_reason", "") or "")
        if control and not reuse_reason and control not in already_independent:
            independent_support_events_by_key[key] = (
                independent_support_events_by_key.get(key, 0) + 1
            )
            support_controls[key].append(control)
            if support > 1:
                reused_control_support_events_by_key[key] = (
                    reused_control_support_events_by_key.get(key, 0) + support - 1
                )
        else:
            reused_control_support_events_by_key[key] = (
                reused_control_support_events_by_key.get(key, 0) + support
            )

    return {
        "tested_hypothesis_keys": tested,
        "support_events_by_key": support_events_by_key,
        "independent_support_events_by_key": independent_support_events_by_key,
        "reused_control_support_events_by_key": reused_control_support_events_by_key,
        "contradiction_events_by_key": contradiction_events_by_key,
        "controlled_experiments_by_key": controlled_experiments_by_key,
        "controls_used_by_key": {
            key: tuple(value) for key, value in controls_used.items()
        },
        "support_controls_by_key": {
            key: tuple(value) for key, value in support_controls.items()
        },
    }


def _experiments_from_controlled_payload(
    payload: Mapping[str, Any],
    *,
    game_id: str | None,
) -> Tuple[Dict[str, Any], ...]:
    detailed = _dict_rows(payload.get("controlled_experiments", []) or [])
    if detailed:
        return tuple(
            row for row in detailed if game_id is None or row.get("game_id") == game_id
        )
    fallback = []
    for row in _dict_rows(payload.get("updated_ledger_entries", []) or []):
        if game_id is not None and row.get("game_id") != game_id:
            continue
        fallback.append(
            {
                "hypothesis_key": row.get("key", ""),
                "game_id": row.get("game_id", ""),
                "support_events": int(row.get("support_events", 0) or 0),
                "contradiction_events": int(row.get("contradiction_events", 0) or 0),
                "controlled_experiments_run": int(
                    row.get("controlled_experiments_run", 0) or 0
                ),
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "contradictions": 0,
            }
        )
    return tuple(fallback)


def _ledger_entries(
    payload: Mapping[str, Any],
    *,
    game_id: str | None,
) -> Tuple[Dict[str, Any], ...]:
    entries = _dict_rows(payload.get("ledger_entries", []) or [])
    if game_id is None:
        return tuple(entries)
    return tuple(entry for entry in entries if str(entry.get("game_id", "")) == game_id)


def _experiment_key(experiment: Mapping[str, Any]) -> str:
    return str(
        experiment.get("hypothesis_key")
        or experiment.get("key")
        or experiment.get("ledger_key")
        or ""
    )


def _dict_rows(rows: Any) -> list[Dict[str, Any]]:
    if not rows:
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _dedupe_dict_rows(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    result: list[Dict[str, Any]] = []
    for row in rows:
        data = {str(key): str(value) for key, value in dict(row).items()}
        signature = tuple(sorted(data.items()))
        if signature in seen:
            continue
        seen.add(signature)
        result.append(dict(row))
    return tuple(result)


def _dedupe_strings(values: Sequence[str]) -> Tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return tuple(result)


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
