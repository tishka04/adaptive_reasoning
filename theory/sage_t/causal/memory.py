"""SAGE.T.A40 append-only causal memory with checksum verification."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    TransitionEvidence,
    causal_program_from_dict,
    transition_evidence_from_dict,
    transition_evidence_to_dict,
)
from .posterior import CausalParticle, CausalPosterior, PosteriorUpdate

MEMORY_FORMAT = "sage-t-causal-memory-v1"


@dataclass(frozen=True)
class CausalMemoryRecord:
    payload: Mapping[str, Any]
    checksum: str


class CausalMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def consolidate(
        self,
        *,
        update: PosteriorUpdate,
        posterior: CausalPosterior,
        evidence: TransitionEvidence,
        maximum_programs: int = 16,
    ) -> CausalMemoryRecord:
        programs = [
            {
                "program": particle.program.to_dict(),
                "log_prior": particle.log_prior,
                "log_weight": particle.log_weight,
                "lineage": list(particle.lineage),
                "evidence_ids": list(particle.evidence_ids),
                "latest_log_likelihood": particle.latest_log_likelihood,
            }
            for particle in posterior.top(maximum_programs)
        ]
        payload = {
            "format_version": MEMORY_FORMAT,
            "record_type": "posterior_update",
            "evidence_id": evidence.evidence_id,
            "game_id": evidence.game_id,
            "context_id": evidence.context_id,
            "prefix_hash": evidence.prefix_hash,
            "terminal": evidence.terminal,
            "success": evidence.success,
            "level_change": evidence.level_change,
            "evidence": transition_evidence_to_dict(evidence),
            "entropy_before": update.entropy_before,
            "entropy_after": update.entropy_after,
            "effective_sample_size": update.effective_sample_size,
            "programs": programs,
        }
        checksum = _checksum(payload)
        record = dict(payload)
        record["checksum"] = checksum
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return CausalMemoryRecord(payload=payload, checksum=checksum)

    def verified_records(self) -> tuple[CausalMemoryRecord, ...]:
        if not self.path.exists():
            return ()
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                checksum = str(raw.pop("checksum", ""))
                if raw.get("format_version") != MEMORY_FORMAT:
                    raise ValueError(f"unsupported causal-memory record at line {line_number}")
                if checksum != _checksum(raw):
                    raise ValueError(f"causal-memory checksum mismatch at line {line_number}")
                records.append(CausalMemoryRecord(payload=raw, checksum=checksum))
        return tuple(records)

    def reload(self, posterior: CausalPosterior) -> int:
        records = self.verified_records()
        if not records:
            return 0
        latest = records[-1].payload
        particles = []
        for item in latest.get("programs", ()):  # type: ignore[union-attr]
            program = causal_program_from_dict(item["program"])
            particles.append(
                CausalParticle(
                    program=program,
                    log_prior=float(item["log_prior"]),
                    log_weight=float(item["log_weight"]),
                    lineage=tuple(item.get("lineage", ())),
                    evidence_ids=tuple(item.get("evidence_ids", ())),
                    latest_log_likelihood=float(item.get("latest_log_likelihood", 0.0)),
                )
            )
        evidence_history = tuple(
            transition_evidence_from_dict(record.payload["evidence"])
            for record in records
            if "evidence" in record.payload
        )
        posterior.restore(particles, evidence=evidence_history)
        return len(particles)

    def promotable_mechanism_ids(
        self,
        *,
        minimum_games: int = 2,
        minimum_contexts_per_game: int = 2,
        minimum_particle_probability: float = 0.5,
    ) -> tuple[str, ...]:
        support: dict[str, dict[str, set[str]]] = {}
        for record in self.verified_records():
            game_id = str(record.payload.get("game_id", ""))
            context_id = str(record.payload.get("context_id", ""))
            if not game_id or not context_id or record.payload.get("terminal") and not record.payload.get("success"):
                continue
            for item in record.payload.get("programs", ()):  # type: ignore[union-attr]
                probability = math.exp(float(item["log_weight"]))
                if probability < minimum_particle_probability:
                    continue
                for mechanism in item["program"].get("mechanisms", ()):
                    module_id = mechanism.get("neural_module_id") or mechanism.get("operator_type")
                    support.setdefault(str(module_id), {}).setdefault(game_id, set()).add(context_id)
        promoted = []
        for mechanism_id, games in support.items():
            qualifying = [
                game for game, contexts in games.items()
                if len(contexts) >= minimum_contexts_per_game
            ]
            if len(qualifying) >= minimum_games:
                promoted.append(mechanism_id)
        return tuple(sorted(promoted))


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["MEMORY_FORMAT", "CausalMemoryRecord", "CausalMemoryStore"]
