"""Goal-relative causal progress programs for SAGE.T12.5.

The key modelling decision is that progress is not a property of a frame.  It
is the state of a small causal automaton after a history of typed effects.  A
progress program is stored beside (and bound to) a complete causal-world
program, so dynamics and goal semantics remain in the same posterior particle.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .option_contracts import EffectAtom, StepEffectContract

PROGRESS_PROGRAM_FORMAT = "sage-t12.5-causal-progress-program-v1"
PROGRESS_POSTERIOR_FORMAT = "sage-t12.5-joint-progress-posterior-v1"
PROGRESS_KINDS = frozenset(
    {"ordered_effects", "unordered_effects", "change_count", "terminal_only"}
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _checksum(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProgressMilestone:
    """One identity-free effect that can advance a goal-progress automaton."""

    atoms: tuple[EffectAtom, ...]
    source_contract_checksum: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", tuple(self.atoms))
        if not self.atoms:
            raise ValueError("a progress milestone needs at least one effect atom")
        if not self.source_contract_checksum:
            raise ValueError("a progress milestone must bind its source contract")

    @property
    def semantic_payload(self) -> dict[str, Any]:
        return {"atoms": [item.to_dict() for item in self.atoms]}

    @property
    def milestone_id(self) -> str:
        return f"milestone_{_checksum(self.semantic_payload)[:16]}"

    def matches(self, step: Mapping[str, Any]) -> bool:
        # Deliberately ignore action labels, absolute positions and identities.
        return all(atom.matches(step) for atom in self.atoms)

    def as_step(self, *, position: int = 0, action_name: str = "ACTION0") -> dict[str, Any]:
        families: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
        for atom in self.atoms:
            before = 0
            families[atom.family][atom.key] = {
                "before": before,
                "after": before + atom.expected_delta,
            }
        return {
            "action_name": str(action_name).upper(),
            "delta": {"mechanism": dict(families)},
            "position": int(position),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.semantic_payload,
            "milestone_id": self.milestone_id,
            "source_contract_checksum": self.source_contract_checksum,
        }

    @classmethod
    def from_contract(cls, contract: StepEffectContract) -> ProgressMilestone:
        return cls(
            atoms=contract.atoms,
            source_contract_checksum=contract.checksum,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProgressMilestone:
        milestone = cls(
            atoms=tuple(
                EffectAtom.from_dict(dict(item)) for item in payload.get("atoms", ())
            ),
            source_contract_checksum=str(payload["source_contract_checksum"]),
        )
        if payload.get("milestone_id") not in {None, milestone.milestone_id}:
            raise ValueError("progress milestone checksum mismatch")
        return milestone


@dataclass(frozen=True)
class CausalProgressProgram:
    """A rival causal explanation of how typed effects approach a goal."""

    progress_kind: str
    owner_program_hash: str
    milestones: tuple[ProgressMilestone, ...]
    goal_predicate: str
    failure_predicate: str | None
    description_length: float
    evidence_ids: tuple[str, ...]
    format_version: str = PROGRESS_PROGRAM_FORMAT

    def __post_init__(self) -> None:
        object.__setattr__(self, "progress_kind", str(self.progress_kind))
        object.__setattr__(self, "milestones", tuple(self.milestones))
        object.__setattr__(self, "evidence_ids", tuple(map(str, self.evidence_ids)))
        object.__setattr__(self, "description_length", float(self.description_length))
        if self.format_version != PROGRESS_PROGRAM_FORMAT:
            raise ValueError("unsupported causal-progress program")
        if self.progress_kind not in PROGRESS_KINDS:
            raise ValueError(f"unsupported progress kind: {self.progress_kind}")
        if not self.owner_program_hash or not self.goal_predicate:
            raise ValueError("progress program needs an owner and a goal")
        if self.progress_kind != "terminal_only" and not self.milestones:
            raise ValueError("non-terminal progress programs need milestones")
        if self.description_length < 0.0 or not math.isfinite(self.description_length):
            raise ValueError("invalid progress-program description length")

    @property
    def safe_payload(self) -> dict[str, Any]:
        return {
            "description_length": self.description_length,
            "evidence_ids": list(self.evidence_ids),
            "failure_predicate": self.failure_predicate,
            "format_version": self.format_version,
            "goal_predicate": self.goal_predicate,
            "milestones": [item.to_dict() for item in self.milestones],
            "owner_program_hash": self.owner_program_hash,
            "progress_kind": self.progress_kind,
        }

    @property
    def canonical_hash(self) -> str:
        return _checksum(self.safe_payload)

    @property
    def program_id(self) -> str:
        return f"progress.{self.progress_kind}.{self.canonical_hash[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.safe_payload,
            "canonical_hash": self.canonical_hash,
            "program_id": self.program_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CausalProgressProgram:
        program = cls(
            progress_kind=str(payload["progress_kind"]),
            owner_program_hash=str(payload["owner_program_hash"]),
            milestones=tuple(
                ProgressMilestone.from_dict(dict(item))
                for item in payload.get("milestones", ())
            ),
            goal_predicate=str(payload["goal_predicate"]),
            failure_predicate=(
                None
                if payload.get("failure_predicate") is None
                else str(payload["failure_predicate"])
            ),
            description_length=float(payload["description_length"]),
            evidence_ids=tuple(map(str, payload.get("evidence_ids", ()))),
            format_version=str(payload.get("format_version", "")),
        )
        if payload.get("canonical_hash") not in {None, program.canonical_hash}:
            raise ValueError("causal-progress program checksum mismatch")
        if payload.get("program_id") not in {None, program.program_id}:
            raise ValueError("causal-progress program id mismatch")
        return program


@dataclass(frozen=True)
class ProgressBelief:
    stage: int
    completed_indices: tuple[int, ...] = ()
    terminal_success_observed: bool = False


@dataclass(frozen=True)
class ProgressTraceEvaluation:
    potentials: tuple[float, ...]
    stages: tuple[int, ...]
    predicted_success: bool
    final_potential: float


class CausalProgressExecutor:
    """Execute progress programs over observed or predicted structured deltas."""

    @staticmethod
    def initial(program: CausalProgressProgram) -> ProgressBelief:
        del program
        return ProgressBelief(stage=0)

    @staticmethod
    def potential(program: CausalProgressProgram, belief: ProgressBelief) -> float:
        if program.progress_kind == "terminal_only":
            return 1.0 if belief.terminal_success_observed else 0.0
        denominator = max(1, len(program.milestones))
        if program.progress_kind == "unordered_effects":
            return len(belief.completed_indices) / denominator
        return min(1.0, belief.stage / denominator)

    def advance(
        self,
        program: CausalProgressProgram,
        belief: ProgressBelief,
        step: Mapping[str, Any],
        *,
        terminal_success: bool = False,
    ) -> ProgressBelief:
        if program.progress_kind == "terminal_only":
            return replace(
                belief,
                stage=1 if terminal_success else belief.stage,
                terminal_success_observed=(
                    belief.terminal_success_observed or bool(terminal_success)
                ),
            )

        stage = belief.stage
        completed = set(belief.completed_indices)
        if program.progress_kind == "ordered_effects":
            if stage < len(program.milestones) and program.milestones[stage].matches(step):
                stage += 1
        elif program.progress_kind == "unordered_effects":
            for index, milestone in enumerate(program.milestones):
                if index not in completed and milestone.matches(step):
                    completed.add(index)
                    break
            stage = len(completed)
        else:  # change_count
            if any(item.matches(step) for item in program.milestones):
                stage = min(len(program.milestones), stage + 1)
        return ProgressBelief(
            stage=stage,
            completed_indices=tuple(sorted(completed)),
            terminal_success_observed=(
                belief.terminal_success_observed or bool(terminal_success)
            ),
        )

    def evaluate_trace(
        self,
        program: CausalProgressProgram,
        steps: Sequence[Mapping[str, Any]],
        *,
        reveal_terminal_success: bool = False,
    ) -> ProgressTraceEvaluation:
        belief = self.initial(program)
        stages = []
        potentials = []
        for index, step in enumerate(steps):
            belief = self.advance(
                program,
                belief,
                step,
                terminal_success=(reveal_terminal_success and index == len(steps) - 1),
            )
            stages.append(belief.stage)
            potentials.append(self.potential(program, belief))
        final = self.potential(program, belief)
        return ProgressTraceEvaluation(
            potentials=tuple(potentials),
            stages=tuple(stages),
            predicted_success=final >= 1.0 - 1e-12,
            final_potential=final,
        )


@dataclass(frozen=True)
class ProgressEvidence:
    evidence_id: str
    lineage_id: str
    steps: tuple[Mapping[str, Any], ...]
    progressed: bool
    modality: str
    action_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(dict(item) for item in self.steps))
        object.__setattr__(self, "action_names", tuple(map(str, self.action_names)))
        if not self.evidence_id or not self.lineage_id or not self.modality:
            raise ValueError("progress evidence needs identity, lineage and modality")


@dataclass
class JointProgressParticle:
    """A lightweight joint reference to one full world program and one goal model."""

    owner_program_hash: str
    progress_program: CausalProgressProgram
    log_weight: float
    lineage: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    @property
    def particle_id(self) -> str:
        return _checksum(
            {
                "owner_program_hash": self.owner_program_hash,
                "progress_program_hash": self.progress_program.canonical_hash,
            }
        )


class JointCausalProgressPosterior:
    """Common posterior over complete world-program + progress-program pairs."""

    def __init__(
        self,
        particles: Sequence[JointProgressParticle],
        *,
        match_probability: float = 0.95,
    ) -> None:
        self.particles = list(particles)
        self.match_probability = float(match_probability)
        self.executor = CausalProgressExecutor()
        if not self.particles:
            raise ValueError("joint progress posterior cannot be empty")
        if not 0.5 < self.match_probability < 1.0:
            raise ValueError("match probability must be in (0.5, 1)")

    @classmethod
    def from_factorized_prior(
        cls,
        *,
        owner_probabilities: Mapping[str, float],
        progress_programs: Sequence[CausalProgressProgram],
        mdl_beta: float = 0.08,
        match_probability: float = 0.95,
    ) -> JointCausalProgressPosterior:
        owners = {
            str(key): max(0.0, float(value))
            for key, value in owner_probabilities.items()
        }
        owner_total = sum(owners.values())
        if owner_total <= 0.0:
            raise ValueError("owner-program prior has no mass")
        by_owner: dict[str, dict[str, CausalProgressProgram]] = defaultdict(dict)
        for program in progress_programs:
            by_owner[program.owner_program_hash].setdefault(
                program.progress_kind, program
            )
        if set(by_owner) != set(owners) or any(
            set(kinds) != set(PROGRESS_KINDS) for kinds in by_owner.values()
        ):
            raise ValueError(
                "joint posterior needs every rival progress kind for every owner"
            )
        particles = []
        for owner_hash, owner_mass in owners.items():
            programs = by_owner[owner_hash]
            raw_progress = {
                kind: math.exp(-float(mdl_beta) * program.description_length)
                for kind, program in programs.items()
            }
            progress_total = sum(raw_progress.values())
            for program in programs.values():
                probability = (owner_mass / owner_total) * (
                    raw_progress[program.progress_kind] / progress_total
                )
                particles.append(
                    JointProgressParticle(
                        owner_program_hash=owner_hash,
                        progress_program=program,
                        log_weight=math.log(max(probability, 1e-300)),
                        lineage=("t12_5:factorized_prior",),
                    )
                )
        return cls(particles, match_probability=match_probability)

    def probabilities(self) -> tuple[float, ...]:
        maximum = max(item.log_weight for item in self.particles)
        raw = [math.exp(item.log_weight - maximum) for item in self.particles]
        total = sum(raw)
        return tuple(value / total for value in raw)

    def update(self, evidence: ProgressEvidence) -> None:
        good = math.log(self.match_probability)
        bad = math.log(1.0 - self.match_probability)
        for particle in self.particles:
            evaluation = self.executor.evaluate_trace(
                particle.progress_program, evidence.steps
            )
            agrees = evaluation.predicted_success is evidence.progressed
            particle.log_weight += good if agrees else bad
            particle.evidence_ids = (*particle.evidence_ids, evidence.evidence_id)

    def update_many(self, evidence: Sequence[ProgressEvidence]) -> None:
        for item in evidence:
            self.update(item)

    def mass_by_kind(self) -> dict[str, float]:
        result = {kind: 0.0 for kind in sorted(PROGRESS_KINDS)}
        for particle, probability in zip(self.particles, self.probabilities()):
            result[particle.progress_program.progress_kind] += probability
        return result

    def mass_by_owner(self) -> dict[str, float]:
        result: dict[str, float] = defaultdict(float)
        for particle, probability in zip(self.particles, self.probabilities()):
            result[particle.owner_program_hash] += probability
        return dict(result)

    def expected_potential(self, steps: Sequence[Mapping[str, Any]]) -> float:
        return sum(
            probability
            * self.executor.evaluate_trace(particle.progress_program, steps).final_potential
            for particle, probability in zip(self.particles, self.probabilities())
        )

    def success_probability(self, steps: Sequence[Mapping[str, Any]]) -> float:
        return sum(
            probability
            for particle, probability in zip(self.particles, self.probabilities())
            if self.executor.evaluate_trace(
                particle.progress_program, steps
            ).predicted_success
        )

    def snapshot(self) -> dict[str, Any]:
        probabilities = self.probabilities()
        return {
            "format_version": PROGRESS_POSTERIOR_FORMAT,
            "joint_particle_count": len(self.particles),
            "mass_by_kind": self.mass_by_kind(),
            "mass_by_owner": self.mass_by_owner(),
            "particles": [
                {
                    "evidence_ids": list(particle.evidence_ids),
                    "log_weight": particle.log_weight,
                    "owner_program_hash": particle.owner_program_hash,
                    "particle_id": particle.particle_id,
                    "probability": probability,
                    "progress_program": particle.progress_program.to_dict(),
                }
                for particle, probability in zip(self.particles, probabilities)
            ],
        }


class CausalProgressActionEvaluator:
    """Score predicted action traces by posterior expected goal proximity."""

    @staticmethod
    def rank(
        posterior: JointCausalProgressPosterior,
        candidates: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        prefix: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[tuple[str, float], ...]:
        scored = [
            (str(name), posterior.expected_potential((*prefix, *tuple(trace))))
            for name, trace in candidates.items()
        ]
        return tuple(sorted(scored, key=lambda item: (-item[1], item[0])))


def rival_progress_programs(
    *,
    owner_program_hash: str,
    effect_contracts: Sequence[StepEffectContract],
    goal_predicate: str,
    failure_predicate: str | None,
    evidence_ids: Sequence[str],
) -> tuple[CausalProgressProgram, ...]:
    milestones = tuple(ProgressMilestone.from_contract(item) for item in effect_contracts)
    lengths = {
        "terminal_only": 1.0,
        "change_count": 2.0 + 0.25 * len(milestones),
        "unordered_effects": 3.0 + 0.5 * len(milestones),
        "ordered_effects": 4.0 + 0.5 * len(milestones),
    }
    return tuple(
        CausalProgressProgram(
            progress_kind=kind,
            owner_program_hash=owner_program_hash,
            milestones=() if kind == "terminal_only" else milestones,
            goal_predicate=goal_predicate,
            failure_predicate=failure_predicate,
            description_length=lengths[kind],
            evidence_ids=tuple(map(str, evidence_ids)),
        )
        for kind in (
            "terminal_only",
            "change_count",
            "unordered_effects",
            "ordered_effects",
        )
    )


__all__ = [
    "CausalProgressActionEvaluator",
    "CausalProgressExecutor",
    "CausalProgressProgram",
    "JointCausalProgressPosterior",
    "JointProgressParticle",
    "ProgressBelief",
    "ProgressEvidence",
    "ProgressMilestone",
    "ProgressTraceEvaluation",
    "rival_progress_programs",
]
