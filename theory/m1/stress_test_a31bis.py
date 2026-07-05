"""M1.4 A31bis stress-test wrapper.

Runs A31 with the M1.3 predicate generator enabled, then writes a compact
baseline-vs-M1 comparison. This module does not add new mechanics; it only
switches the optional predicate generator on.

Run:
    ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.stress_test_a31bis --budgets 5,10,25,50 --latest-per-game
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from theory.multi_game_stress_test import (
    DEFAULT_STRESS_BUDGETS,
    DEFAULT_TRACES_DIR,
    classify_failure,
    run_multi_game_stress_test,
    select_stress_traces,
)

from .invariant_miner import DEFAULT_ACCEPTED_PATH
from .anchor_expansion import build_m1_anchor_expander
from .live_anchor_ranking import (
    M1_LIVE_PREFERRED_PREDICATES,
    build_m1_live_candidate_ranker,
    run_live_anchor_compatibility_pretest,
)
from .predicate_generation import (
    DEFAULT_PRETEST_TOP_K,
    build_m1_predicate_generator,
    run_predicate_coverage_pretest,
)

DEFAULT_BASELINE_PATH = Path("diagnostics") / "multi_game_stress_test.json"
DEFAULT_OUTPUT_PATH = Path("diagnostics") / "m1_stress_test_a31bis.json"


@dataclass(frozen=True)
class A31BisSummary:
    """Compact metrics for one stress-test result."""

    label: str
    trace_count: int
    game_count: int
    experiments_run: int
    hypotheses_confirmed: int
    hypotheses_refuted: int
    useful_new_states: int
    wrong_confirmations: int
    failure_types: Dict[str, int]

    @property
    def not_enough_relation_candidates(self) -> int:
        return int(self.failure_types.get("not_enough_relation_candidates", 0))

    @property
    def confirmations_per_experiment(self) -> float:
        return _safe_ratio(self.hypotheses_confirmed, self.experiments_run)

    @property
    def refutations_per_experiment(self) -> float:
        return _safe_ratio(self.hypotheses_refuted, self.experiments_run)

    @property
    def useful_new_states_per_experiment(self) -> float:
        return _safe_ratio(self.useful_new_states, self.experiments_run)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "trace_count": self.trace_count,
            "game_count": self.game_count,
            "experiments_run": self.experiments_run,
            "hypotheses_confirmed": self.hypotheses_confirmed,
            "hypotheses_refuted": self.hypotheses_refuted,
            "confirmations_per_experiment": round(
                self.confirmations_per_experiment,
                4,
            ),
            "refutations_per_experiment": round(
                self.refutations_per_experiment,
                4,
            ),
            "useful_new_states": self.useful_new_states,
            "useful_new_states_per_experiment": round(
                self.useful_new_states_per_experiment,
                4,
            ),
            "wrong_confirmations": self.wrong_confirmations,
            "not_enough_relation_candidates": self.not_enough_relation_candidates,
            "failure_types": dict(sorted(self.failure_types.items())),
        }


def run_a31bis(
    *,
    traces_dir: Path | str = DEFAULT_TRACES_DIR,
    trace_paths: Sequence[Path | str] = (),
    budgets: Sequence[int] = DEFAULT_STRESS_BUDGETS,
    latest_per_game: bool = True,
    max_traces: int = 0,
    include_ar25: bool = True,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    run_memory_comparison: bool = True,
    max_attempts_per_budget: int | None = None,
    stagnation_limit: int = 2,
    ar25_revise_every: int = 20,
    accepted_invariants_path: Path | str = DEFAULT_ACCEPTED_PATH,
    baseline_path: Path | str = DEFAULT_BASELINE_PATH,
    run_local_baseline: bool = False,
    use_anchor_expansion: bool = True,
    use_live_anchor_ranking: bool = True,
) -> Dict[str, Any]:
    """Run A31bis and return a JSON-ready comparison payload."""
    predicate_generator = build_m1_predicate_generator(
        accepted_invariants_path=accepted_invariants_path,
    )
    anchor_expander = build_m1_anchor_expander() if use_anchor_expansion else None
    candidate_ranker = (
        build_m1_live_candidate_ranker()
        if use_anchor_expansion and use_live_anchor_ranking
        else None
    )
    preferred_predicates = (
        M1_LIVE_PREFERRED_PREDICATES
        if candidate_ranker is not None
        else None
    )
    m1_result = run_multi_game_stress_test(
        traces_dir=traces_dir,
        trace_paths=trace_paths,
        budgets=budgets,
        max_traces=max_traces,
        include_ar25=include_ar25,
        latest_per_game=latest_per_game,
        environments_dir=environments_dir,
        max_candidates=max_candidates,
        min_pixel_support=min_pixel_support,
        run_memory_comparison=run_memory_comparison,
        max_attempts_per_budget=max_attempts_per_budget,
        stagnation_limit=stagnation_limit,
        ar25_revise_every=ar25_revise_every,
        predicate_generator=predicate_generator,
        anchor_expander=anchor_expander,
        candidate_ranker=candidate_ranker,
        preferred_predicates=preferred_predicates,
    )
    local_baseline_result = None
    if run_local_baseline:
        local_baseline_result = run_multi_game_stress_test(
            traces_dir=traces_dir,
            trace_paths=trace_paths,
            budgets=budgets,
            max_traces=max_traces,
            include_ar25=include_ar25,
            latest_per_game=latest_per_game,
            environments_dir=environments_dir,
            max_candidates=max_candidates,
            min_pixel_support=min_pixel_support,
            run_memory_comparison=run_memory_comparison,
            max_attempts_per_budget=max_attempts_per_budget,
            stagnation_limit=stagnation_limit,
            ar25_revise_every=ar25_revise_every,
            predicate_generator=None,
            anchor_expander=None,
            candidate_ranker=None,
            preferred_predicates=None,
        )
    selected_traces = select_stress_traces(
        traces_dir=traces_dir,
        trace_paths=trace_paths,
        max_traces=max_traces,
        include_ar25=include_ar25,
        latest_per_game=latest_per_game,
    )
    non_ar25_trace_paths = [
        trace.trace_path
        for trace in selected_traces
        if not trace.game_id.startswith("ar25")
    ]
    coverage = run_predicate_coverage_pretest(
        non_ar25_trace_paths,
        accepted_invariants_path=accepted_invariants_path,
        min_pixel_support=min_pixel_support,
        top_k=max(DEFAULT_PRETEST_TOP_K, int(max_candidates)),
        anchor_expander=anchor_expander,
    )
    live_compatibility = (
        run_live_anchor_compatibility_pretest(
            non_ar25_trace_paths,
            accepted_invariants_path=accepted_invariants_path,
            environments_dir=environments_dir,
            min_pixel_support=min_pixel_support,
            max_candidates=max_candidates,
            discovery_top_k=max(DEFAULT_PRETEST_TOP_K, int(max_candidates) * 5),
        )
        if candidate_ranker is not None
        else None
    )
    baseline_raw = load_baseline_json(baseline_path)
    archived_baseline_summary = summarize_stress_dict(
        baseline_raw,
        label="baseline_archived",
    )
    baseline_summary = (
        summarize_stress_dict(
            local_baseline_result.to_dict(),
            label="baseline_local_same_config",
        )
        if local_baseline_result is not None
        else archived_baseline_summary
    )
    m1_summary = summarize_stress_dict(m1_result.to_dict(), label="m1_a31bis")
    return {
        "config": {
            "budgets": [int(value) for value in budgets],
            "latest_per_game": bool(latest_per_game),
            "max_traces": int(max_traces),
            "include_ar25": bool(include_ar25),
            "max_candidates": int(max_candidates),
            "min_pixel_support": int(min_pixel_support),
            "run_memory_comparison": bool(run_memory_comparison),
            "accepted_invariants_path": str(accepted_invariants_path),
            "baseline_path": str(baseline_path),
            "run_local_baseline": bool(run_local_baseline),
            "m1_anchor_expansion": bool(use_anchor_expansion),
            "m1_live_anchor_ranking": bool(candidate_ranker is not None),
        },
        "baseline": baseline_summary.to_dict(),
        "archived_baseline": archived_baseline_summary.to_dict(),
        "m1": m1_summary.to_dict(),
        "deltas": comparison_deltas(baseline_summary, m1_summary),
        "coverage_pretest": coverage.to_dict(),
        "m1_stress_result": m1_result.to_dict(),
        "local_baseline_stress_result": (
            local_baseline_result.to_dict()
            if local_baseline_result is not None
            else None
        ),
        "live_anchor_compatibility_pretest": (
            live_compatibility.to_dict() if live_compatibility is not None else None
        ),
        "diagnosis": diagnose_result(
            baseline_summary,
            m1_summary,
            coverage.to_dict(),
            live_compatibility.to_dict() if live_compatibility is not None else None,
        ),
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def summarize_stress_dict(payload: Mapping[str, Any], *, label: str) -> A31BisSummary:
    return A31BisSummary(
        label=label,
        trace_count=int(payload.get("trace_count", 0) or 0),
        game_count=int(payload.get("game_count", 0) or 0),
        experiments_run=int(payload.get("experiments_run", 0) or 0),
        hypotheses_confirmed=int(payload.get("hypotheses_confirmed", 0) or 0),
        hypotheses_refuted=int(payload.get("hypotheses_refuted", 0) or 0),
        useful_new_states=int(payload.get("useful_new_states", 0) or 0),
        wrong_confirmations=int(payload.get("wrong_confirmations", 0) or 0),
        failure_types=_normalized_failure_types(payload),
    )


def comparison_deltas(
    baseline: A31BisSummary,
    m1: A31BisSummary,
) -> Dict[str, Any]:
    return {
        "not_enough_relation_candidates": (
            m1.not_enough_relation_candidates
            - baseline.not_enough_relation_candidates
        ),
        "wrong_confirmations": m1.wrong_confirmations - baseline.wrong_confirmations,
        "experiments_run": m1.experiments_run - baseline.experiments_run,
        "hypotheses_confirmed": (
            m1.hypotheses_confirmed - baseline.hypotheses_confirmed
        ),
        "hypotheses_refuted": m1.hypotheses_refuted - baseline.hypotheses_refuted,
        "useful_new_states": m1.useful_new_states - baseline.useful_new_states,
        "confirmations_per_experiment": round(
            m1.confirmations_per_experiment
            - baseline.confirmations_per_experiment,
            4,
        ),
        "refutations_per_experiment": round(
            m1.refutations_per_experiment - baseline.refutations_per_experiment,
            4,
        ),
        "useful_new_states_per_experiment": round(
            m1.useful_new_states_per_experiment
            - baseline.useful_new_states_per_experiment,
            4,
        ),
    }


def diagnose_result(
    baseline: A31BisSummary,
    m1: A31BisSummary,
    coverage: Mapping[str, Any],
    live_compatibility: Mapping[str, Any] | None = None,
) -> str:
    averages = coverage.get("averages", {})
    unique_delta = _nested_number(
        averages,
        "unique_predicates_per_trace",
        "delta",
    )
    pairs_delta = _nested_number(
        averages,
        "candidate_pairs_per_trace",
        "delta",
    )
    generated_delta = _nested_number(
        averages,
        "relation_candidates_generated",
        "delta",
    )
    blocker_delta = (
        m1.not_enough_relation_candidates
        - baseline.not_enough_relation_candidates
    )
    if live_compatibility is not None:
        live_after = live_compatibility.get("averages_after_ranking", {})
        entering_agenda = _nested_number(
            live_after,
            "new_pairs_entering_agenda",
        )
        live_compatible = _nested_number(
            live_after,
            "new_pairs_live_color_compatible",
        )
        if entering_agenda > 0 and blocker_delta < 0:
            return "live_anchor_ranking_blocker_reduced"
        if entering_agenda > 0 and blocker_delta >= 0:
            return "live_anchor_ranking_entering_agenda_blocker_not_reduced"
        if live_compatible > 0 and blocker_delta >= 0:
            return "live_anchor_ranking_no_new_agenda_pairs"
    if unique_delta > 0 and generated_delta > 0 and pairs_delta <= 0 and blocker_delta >= 0:
        return "vocabulary_expanded_pairs_flat_blocker_not_reduced"
    if unique_delta > 0 and generated_delta > 0 and pairs_delta > 0 and blocker_delta < 0:
        return "anchor_expansion_blocker_reduced"
    if unique_delta > 0 and generated_delta > 0 and pairs_delta > 0 and blocker_delta >= 0:
        return "anchor_expansion_pairs_up_blocker_not_reduced"
    if unique_delta > 0 and generated_delta > 0 and blocker_delta < 0:
        return "vocabulary_expanded_blocker_reduced"
    if unique_delta <= 0:
        return "vocabulary_not_expanded"
    if generated_delta > 0 and blocker_delta >= 0:
        return "candidates_generated_but_blocker_not_reduced"
    return "mixed"


def load_baseline_json(path: Path | str) -> Dict[str, Any]:
    text = _read_text_with_fallback(Path(path))
    start = text.find("{")
    if start < 0:
        return {}
    return json.loads(text[start:])


def write_a31bis_result(
    payload: Mapping[str, Any],
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalized_failure_types(payload: Mapping[str, Any]) -> Dict[str, int]:
    raw = payload.get("failure_types") or {}
    if isinstance(raw, Mapping):
        return {str(key): int(value) for key, value in raw.items()}
    counts: Counter[str] = Counter()
    for point in payload.get("curve_points", []) or []:
        for error in point.get("errors", []) or []:
            kind = classify_failure(error)
            if kind:
                counts[kind] += 1
    return dict(counts)


def _nested_number(payload: Mapping[str, Any], *keys: str) -> float:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return 0.0
        value = value.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text()


def _parse_budgets(raw: str) -> List[int]:
    return [
        int(item.strip())
        for item in str(raw).split(",")
        if item.strip()
    ]


def _parse_paths(values: Sequence[str]) -> List[Path]:
    return [Path(value) for value in values if value]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M1.4 A31bis stress test.")
    parser.add_argument("--traces-dir", type=Path, default=DEFAULT_TRACES_DIR)
    parser.add_argument("--trace-path", action="append", default=[])
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_STRESS_BUDGETS)))
    parser.add_argument("--latest-per-game", action="store_true")
    parser.add_argument("--max-traces", type=int, default=0)
    parser.add_argument("--exclude-ar25", action="store_true")
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    parser.add_argument("--skip-memory-comparison", action="store_true")
    parser.add_argument("--max-attempts-per-budget", type=int, default=None)
    parser.add_argument("--stagnation-limit", type=int, default=2)
    parser.add_argument("--ar25-revise-every", type=int, default=20)
    parser.add_argument("--accepted-invariants", type=Path, default=DEFAULT_ACCEPTED_PATH)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--run-local-baseline", action="store_true")
    parser.add_argument("--disable-m1-anchor-expansion", action="store_true")
    parser.add_argument("--disable-m1-live-anchor-ranking", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_a31bis(
        traces_dir=args.traces_dir,
        trace_paths=_parse_paths(args.trace_path),
        budgets=_parse_budgets(args.budgets),
        latest_per_game=args.latest_per_game,
        max_traces=args.max_traces,
        include_ar25=not args.exclude_ar25,
        environments_dir=args.environments_dir,
        max_candidates=args.max_candidates,
        min_pixel_support=args.min_pixel_support,
        run_memory_comparison=not args.skip_memory_comparison,
        max_attempts_per_budget=args.max_attempts_per_budget,
        stagnation_limit=args.stagnation_limit,
        ar25_revise_every=args.ar25_revise_every,
        accepted_invariants_path=args.accepted_invariants,
        baseline_path=args.baseline,
        run_local_baseline=args.run_local_baseline,
        use_anchor_expansion=not args.disable_m1_anchor_expansion,
        use_live_anchor_ranking=not args.disable_m1_live_anchor_ranking,
    )
    write_a31bis_result(payload, args.out)
    summary = {
        "output_path": str(args.out),
        "baseline_not_enough_relation_candidates": payload["baseline"][
            "not_enough_relation_candidates"
        ],
        "m1_not_enough_relation_candidates": payload["m1"][
            "not_enough_relation_candidates"
        ],
        "m1_wrong_confirmations": payload["m1"]["wrong_confirmations"],
        "coverage_unique_predicates": payload["coverage_pretest"]["averages"][
            "unique_predicates_per_trace"
        ],
        "coverage_relation_candidates_generated": payload["coverage_pretest"][
            "averages"
        ]["relation_candidates_generated"],
        "coverage_candidate_pairs": payload["coverage_pretest"]["averages"][
            "candidate_pairs_per_trace"
        ],
        "live_anchor_compatibility": (
            None
            if payload["live_anchor_compatibility_pretest"] is None
            else payload["live_anchor_compatibility_pretest"]["averages_after_ranking"]
        ),
        "diagnosis": payload["diagnosis"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
