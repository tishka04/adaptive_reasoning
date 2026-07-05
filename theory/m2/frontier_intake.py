"""M2 intake for A40 frontier handoff requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.a40.frontier_handoff_requests import (
    DEFAULT_A40_FRONTIER_HANDOFF_OUTPUT_PATH,
)

from .schema import M2_TRUTH_STATUS


DEFAULT_M2_FRONTIER_INTAKE_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "frontier_intake.json"
)


def run_frontier_intake(
    *,
    frontier_path: str | Path = DEFAULT_A40_FRONTIER_HANDOFF_OUTPUT_PATH,
) -> Dict[str, Any]:
    payload = _load_json(frontier_path)
    return run_frontier_intake_from_payload(
        payload,
        input_frontier_path=str(frontier_path),
    )


def run_frontier_intake_from_payload(
    payload: Mapping[str, Any],
    *,
    input_frontier_path: str,
) -> Dict[str, Any]:
    frontiers = payload.get("frontier_requests", []) or []
    open_frontiers: list[Dict[str, Any]] = []
    invalid_frontiers: list[Dict[str, Any]] = []

    for index, frontier in enumerate(frontiers):
        valid, reason = classify_frontier_request(frontier)
        if valid:
            open_frontiers.append(dict(frontier))
        else:
            invalid_frontiers.append(
                {
                    "index": index,
                    "request_id": (
                        str(frontier.get("request_id", ""))
                        if isinstance(frontier, Mapping)
                        else ""
                    ),
                    "reason": reason,
                }
            )

    summary = {
        "input_frontier_path": input_frontier_path,
        "frontier_requests_consumed": len(open_frontiers),
        "open_frontiers": len(open_frontiers),
        "closed_or_invalid_frontiers": len(invalid_frontiers),
        "ready_for_generation": bool(open_frontiers),
        "truth_status": M2_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }
    return {
        "config": {
            "input_frontier_path": input_frontier_path,
            "inputs_read": ["A40"],
            "artifacts_not_modified": ["A32", "A33", "A34", "A35", "A36", "A37", "A38", "A39", "A40", "M3"],
            "schema_version": "m2.intake.v1",
        },
        "frontier_requests": open_frontiers,
        "invalid_frontiers": invalid_frontiers,
        "summary": summary,
        "truth_status": M2_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def classify_frontier_request(frontier: Any) -> tuple[bool, str]:
    if not isinstance(frontier, Mapping):
        return False, "not_mapping"
    if not bool(frontier.get("ready_for_m1_or_m3", False)):
        return False, "not_ready_for_m1_or_m3"
    if str(frontier.get("status", "OPEN")).upper() != "OPEN":
        return False, "frontier_not_open"
    missing = [
        field
        for field in ("request_id", "game_id", "frontier_context_id", "reason")
        if not str(frontier.get(field, ""))
    ]
    if missing:
        return False, "missing_" + ",".join(missing)
    return True, ""


def write_frontier_intake(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_M2_FRONTIER_INTAKE_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    if not Path(path).exists():
        return {"frontier_requests": []}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M2 A40 frontier intake.")
    parser.add_argument(
        "--frontiers",
        type=Path,
        default=DEFAULT_A40_FRONTIER_HANDOFF_OUTPUT_PATH,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_M2_FRONTIER_INTAKE_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_frontier_intake(frontier_path=args.frontiers)
    write_frontier_intake(payload, args.out)
    print(
        json.dumps(
            {"output_path": str(args.out), "summary": payload["summary"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
