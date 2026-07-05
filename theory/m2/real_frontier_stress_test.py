"""Real A40 frontier corpus smoke/stress runner for M2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .frontier_conditioned_hypotheses import run_frontier_conditioned_hypotheses
from .schema import M2_TRUTH_STATUS


DEFAULT_M2_REAL_FRONTIER_STRESS_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "real_frontier_stress_test.json"
)


def run_real_frontier_stress_test(
    *,
    frontier_paths: Sequence[str | Path] | None = None,
) -> Dict[str, Any]:
    paths = tuple(Path(path) for path in (frontier_paths or _default_frontier_paths()))
    files_consumed = 0
    frontiers_consumed = 0
    hypotheses_generated = 0
    testable = 0
    blocked = 0
    per_file = []
    for path in paths:
        if not path.exists():
            continue
        files_consumed += 1
        outputs = run_frontier_conditioned_hypotheses(frontier_path=path)
        summary = outputs["hypothesis_payload"]["summary"]
        frontiers_consumed += int(summary["frontier_requests_consumed"])
        hypotheses_generated += int(summary["hypotheses_generated"])
        testable += int(summary["testable_hypotheses"])
        blocked += int(summary["blocked_not_testable_hypotheses"])
        per_file.append({"path": str(path), "summary": summary})

    total = max(1, hypotheses_generated)
    summary = {
        "real_frontier_files_consumed": files_consumed,
        "frontier_requests_consumed": frontiers_consumed,
        "hypotheses_generated": hypotheses_generated,
        "testable_ratio": round(testable / total, 4),
        "blocked_not_testable_ratio": round(blocked / total, 4),
        "truth_status": M2_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }
    return {
        "summary": summary,
        "files": per_file,
        "truth_status": M2_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def write_real_frontier_stress_test(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_M2_REAL_FRONTIER_STRESS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _default_frontier_paths() -> tuple[Path, ...]:
    root = Path("diagnostics") / "a40"
    paths = tuple(sorted(root.glob("*frontier*.json")))
    default = root / "frontier_handoff_requests.json"
    if paths:
        return paths
    return (default,)
