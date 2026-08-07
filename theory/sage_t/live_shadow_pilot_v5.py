"""T8.5 live pilot with materialized baseline-action injection.

The unified controller may emit a parameterized action that ARC can
materialize even when the SDK's candidate list contains only representative
parameterizations.  T8.5 adds that already-selected baseline action to the
local counterfactual set and reserves one sequence for it.  Shadow authority
remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)

from . import live_shadow_pilot as base
from .contracts import ActionCandidate, normalized_action_candidates
from .controller import SageTConfig
from .live_shadow_pilot_v3 import assessment_for_live_action
from .live_shadow_pilot_v4 import BaselineInclusiveDecisionEngine

FORMAT_VERSION = "sage-t8.5-live-shadow-materialized-action-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t8_5_frozen_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    Path("training") / "sage_t" / "live_shadow_pilot_v1_t8_5"
)


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
        raise ValueError("SAGE.T8.5 manifest checksum mismatch")
    if payload.get("t8_5_format_version") != FORMAT_VERSION:
        raise ValueError("unsupported SAGE.T8.5 manifest")
    base.load_frozen_manifest(path)
    expected_hash = payload.get("code_sha256", {}).get(
        "live_shadow_pilot_v5.py"
    )
    if not expected_hash:
        raise ValueError("SAGE.T8.5 code hash is missing")
    if _file_sha256(Path(__file__)) != expected_hash:
        raise ValueError("SAGE.T8.5 materialized-action code drifted")
    if payload.get("inference_changes") != [
        "add the materializable baseline action to the local candidate set",
        "reserve one counterfactual sequence for that exact action",
    ]:
        raise ValueError("SAGE.T8.5 contains an unregistered inference change")
    return payload


def _semantic_action_data(value: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(value or {})
    normalized.pop("game_id", None)
    return normalized


def materialized_baseline_candidate(
    *,
    symbolic_action_name: str,
    symbolic_action_data: Mapping[str, Any] | None,
    legal_actions: Sequence[Any],
) -> ActionCandidate | None:
    """Mirror UCC materialization while returning a SAGE.T candidate."""

    name = str(symbolic_action_name).strip().upper()
    data = _semantic_action_data(symbolic_action_data)
    try:
        candidates = normalized_action_candidates(legal_actions)
    except (TypeError, ValueError):
        return None
    same_name = [candidate for candidate in candidates if candidate.action_name == name]
    if not same_name:
        return None
    exact = [
        candidate
        for candidate in same_name
        if _semantic_action_data(candidate.action_data) == data
    ]
    if exact:
        return exact[0]
    if data:
        return ActionCandidate(name, data)
    generic = [candidate for candidate in same_name if not candidate.action_data]
    return generic[0] if generic else same_name[0]


class MaterializedActionController(base.InstrumentedSageTController):
    """Assess the exact safe baseline action, including synthesized parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.decision_engine = BaselineInclusiveDecisionEngine(
            executor=self.executor,
            maximum_sequences=self.config.maximum_sequences,
            maximum_particles=self.config.maximum_particles_per_decision,
            ordinary_horizon=self.config.ordinary_horizon,
        )

    def decide(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        preferred = materialized_baseline_candidate(
            symbolic_action_name=kwargs.get("symbolic_action_name", ""),
            symbolic_action_data=kwargs.get("symbolic_action_data"),
            legal_actions=kwargs.get("legal_actions", ()),
        )
        updated = dict(kwargs)
        if preferred is not None:
            normalized = normalized_action_candidates(
                kwargs.get("legal_actions", ())
            )
            if all(candidate.key != preferred.key for candidate in normalized):
                updated["legal_actions"] = (preferred, *normalized)
        self.decision_engine.preferred_action = preferred
        try:
            return super().decide(**updated)
        finally:
            self.decision_engine.preferred_action = None


def _controller_factory(
    *,
    mode: str,
    manifest: Mapping[str, Any],
) -> Any:
    caps = manifest["controller"]

    def factory(game_id: str) -> UnifiedCognitiveController:
        if mode == "off":
            return UnifiedCognitiveController(
                game_id,
                config=UnifiedCognitiveConfig(sage_t_authority_mode="off"),
            )
        sage_t = MaterializedActionController(
            config=SageTConfig(
                mode="shadow",
                maximum_programs=int(caps["maximum_programs"]),
                maximum_sequences=int(caps["maximum_sequences"]),
                maximum_particles_per_decision=int(
                    caps["maximum_particles_per_decision"]
                ),
                ordinary_horizon=int(caps["ordinary_horizon"]),
            )
        )
        return UnifiedCognitiveController(
            game_id,
            config=UnifiedCognitiveConfig(sage_t_authority_mode="shadow"),
            sage_t_controller=sage_t,
        )

    return factory


def run_live_shadow_pilot(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    environments_dir: str | Path = "environment_files",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    load_frozen_manifest(manifest_path)
    previous_assessment = base._assessment_for_action
    previous_factory = base._controller_factory
    base._assessment_for_action = assessment_for_live_action
    base._controller_factory = _controller_factory
    try:
        return base.run_live_shadow_pilot(
            manifest_path=manifest_path,
            environments_dir=environments_dir,
            output_dir=output_dir,
        )
    finally:
        base._assessment_for_action = previous_assessment
        base._controller_factory = previous_factory


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
    "MaterializedActionController",
    "load_frozen_manifest",
    "main",
    "materialized_baseline_candidate",
    "run_live_shadow_pilot",
]
