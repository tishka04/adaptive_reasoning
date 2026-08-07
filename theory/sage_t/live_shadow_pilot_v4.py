"""T8.4 live pilot with guaranteed executed-action coverage.

The live ARC action space can contain hundreds of parameterized ACTION6 clicks.
T8.2/T8.3 evaluated only the first eight candidate sequences, so the action
selected by the baseline controller was usually absent from the
counterfactual matrix.  T8.4 reserves the first sequence for that exact action
without changing the action executed in shadow mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)

from . import live_shadow_pilot as base
from .contracts import ActionCandidate, normalized_action_candidates
from .controller import SageTConfig
from .decision import CandidateSequence, CounterfactualDecisionEngine
from .live_shadow_pilot_v3 import assessment_for_live_action

FORMAT_VERSION = "sage-t8.4-live-shadow-baseline-inclusive-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t8_4_frozen_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    Path("training") / "sage_t" / "live_shadow_pilot_v1_t8_4"
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
        raise ValueError("SAGE.T8.4 manifest checksum mismatch")
    if payload.get("t8_4_format_version") != FORMAT_VERSION:
        raise ValueError("unsupported SAGE.T8.4 manifest")
    base.load_frozen_manifest(path)
    expected_hash = payload.get("code_sha256", {}).get(
        "live_shadow_pilot_v4.py"
    )
    if not expected_hash:
        raise ValueError("SAGE.T8.4 code hash is missing")
    if _file_sha256(Path(__file__)) != expected_hash:
        raise ValueError("SAGE.T8.4 baseline-inclusion code drifted")
    if payload.get("inference_changes") != [
        "reserve one counterfactual sequence for the exact baseline action"
    ]:
        raise ValueError("SAGE.T8.4 contains an unregistered inference change")
    return payload


@dataclass
class BaselineInclusiveDecisionEngine(CounterfactualDecisionEngine):
    """Place the baseline action first in the bounded sequence matrix."""

    preferred_action: ActionCandidate | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def generate_sequences(
        self,
        legal_actions: Sequence[ActionCandidate],
        *,
        memory_macros: Sequence[Sequence[ActionCandidate]] = (),
    ) -> tuple[CandidateSequence, ...]:
        macros = list(memory_macros)
        if self.preferred_action is not None:
            macros.insert(0, (self.preferred_action,))
        return super().generate_sequences(
            legal_actions,
            memory_macros=tuple(macros),
        )


def _semantic_action_data(value: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(value or {})
    normalized.pop("game_id", None)
    return normalized


def find_baseline_candidate(
    *,
    symbolic_action_name: str,
    symbolic_action_data: Mapping[str, Any] | None,
    legal_actions: Sequence[Any],
) -> ActionCandidate | None:
    """Resolve the baseline action to its exact normalized legal candidate."""

    name = str(symbolic_action_name).strip().upper()
    data = _semantic_action_data(symbolic_action_data)
    try:
        candidates = normalized_action_candidates(legal_actions)
    except (TypeError, ValueError):
        return None
    exact = [
        candidate
        for candidate in candidates
        if candidate.action_name == name
        and _semantic_action_data(candidate.action_data) == data
    ]
    if exact:
        return exact[0]
    generic = [
        candidate
        for candidate in candidates
        if candidate.action_name == name and not candidate.action_data
    ]
    return generic[0] if generic and name != "ACTION6" else None


class BaselineInclusiveController(base.InstrumentedSageTController):
    """Instrumented shadow controller that always assesses the live action."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.decision_engine = BaselineInclusiveDecisionEngine(
            executor=self.executor,
            maximum_sequences=self.config.maximum_sequences,
            maximum_particles=self.config.maximum_particles_per_decision,
            ordinary_horizon=self.config.ordinary_horizon,
        )

    def decide(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        self.decision_engine.preferred_action = find_baseline_candidate(
            symbolic_action_name=kwargs.get("symbolic_action_name", ""),
            symbolic_action_data=kwargs.get("symbolic_action_data"),
            legal_actions=kwargs.get("legal_actions", ()),
        )
        try:
            return super().decide(**kwargs)
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
        sage_t = BaselineInclusiveController(
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
    "BaselineInclusiveController",
    "BaselineInclusiveDecisionEngine",
    "find_baseline_candidate",
    "load_frozen_manifest",
    "main",
    "run_live_shadow_pilot",
]
