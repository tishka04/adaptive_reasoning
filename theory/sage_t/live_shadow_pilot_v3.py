"""T8.3 live pilot with local-action metadata normalization.

T8.3 preserves the T8.2 source-signal panel and controller behavior.  It only
normalizes the non-semantic ``game_id`` field before matching an executed live
action to the abstract counterfactual sequence used for calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import live_shadow_pilot as base

FORMAT_VERSION = "sage-t8.3-live-shadow-normalized-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t8_3_frozen_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "live_shadow_pilot_v1_t8_3"


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _checksum(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != _checksum(unsigned):
        raise ValueError("SAGE.T8.3 manifest checksum mismatch")
    if payload.get("t8_3_format_version") != FORMAT_VERSION:
        raise ValueError("unsupported SAGE.T8.3 manifest")
    base.load_frozen_manifest(path)
    expected_hash = payload.get("code_sha256", {}).get(
        "live_shadow_pilot_v3.py"
    )
    if not expected_hash:
        raise ValueError("SAGE.T8.3 code hash is missing")
    if _file_sha256(Path(__file__)) != expected_hash:
        raise ValueError("SAGE.T8.3 normalization code drifted")
    if payload.get("measurement_changes") != [
        "drop non-semantic game_id before action-key comparison"
    ]:
        raise ValueError("SAGE.T8.3 contains an unregistered measurement change")
    return payload


def _parse_action_key(value: str) -> tuple[str, dict[str, Any]]:
    name, separator, encoded = str(value).partition(":")
    if not separator:
        return name.strip().upper(), {}
    try:
        data = json.loads(encoded)
    except json.JSONDecodeError:
        return name.strip().upper(), {}
    normalized = dict(data) if isinstance(data, dict) else {}
    normalized.pop("game_id", None)
    return name.strip().upper(), normalized


def assessment_for_live_action(
    decision: Mapping[str, Any],
    action_key: str,
) -> Mapping[str, Any] | None:
    executed_name, executed_data = _parse_action_key(action_key)
    exact = []
    same_name = []
    for assessment in decision.get("sequences", ()) or ():
        sequence = tuple(assessment.get("sequence", ()) or ())
        if not sequence:
            continue
        candidate_name, candidate_data = _parse_action_key(sequence[0])
        if candidate_name != executed_name:
            continue
        same_name.append(assessment)
        if candidate_data == executed_data:
            exact.append(assessment)
    matches = exact or (
        same_name if executed_name != "ACTION6" and not executed_data else []
    )
    return min(matches, key=lambda item: len(item["sequence"])) if matches else None


def run_live_shadow_pilot(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    environments_dir: str | Path = "environment_files",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    load_frozen_manifest(manifest_path)
    previous = base._assessment_for_action
    base._assessment_for_action = assessment_for_live_action
    try:
        report = base.run_live_shadow_pilot(
            manifest_path=manifest_path,
            environments_dir=environments_dir,
            output_dir=output_dir,
        )
    finally:
        base._assessment_for_action = previous
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_live_shadow_pilot(
        manifest_path=args.manifest,
        environments_dir=args.environments_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "diagnosis": report.get("diagnosis"),
                "rows": report.get("rows"),
                "prediction_coverage": report.get("metrics", {}).get(
                    "prediction_coverage"
                ),
                "source_validation_authorized": report.get(
                    "source_validation_authorized"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.get("integration_gate_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "assessment_for_live_action",
    "load_frozen_manifest",
    "main",
    "run_live_shadow_pilot",
]
