"""Frozen SAGE.T7 same-prestate scientific replay and gate report.

The replay is deliberately source-only.  Counterfactual outcomes are used only
by the evaluator (held-out likelihood and a grammar-relative oracle family);
they never enter proposal, posterior update, repair, or action selection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

from theory.sage12.bound_mechanic_pilot import BindingPairRecord, load_pairs

from .contracts import (
    AbstractEntity,
    AbstractState,
    ActionCandidate,
    GroundFact,
    ObservedTransition,
    PredictionPacket,
)
from .decision import CounterfactualDecisionEngine
from .evaluation import (
    CounterfactualPanel,
    count_forbidden_program_fields,
)
from .executor import ProgramExecutor
from .posterior import (
    DEFAULT_CHANNEL_WEIGHTS,
    ProgramPosterior,
    packet_log_likelihood,
)
from .synthesis import (
    AssembledProgram,
    DeterministicFragmentProposer,
    ProgramAssembler,
)

FORMAT_VERSION = "sage-t7-scientific-replay-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t7_frozen_manifest.json")
DEFAULT_V43_DIR = (
    Path("training") / "sage12" / "bound_mechanic_pilot_v4_3"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "replay_scientific_v1"
FROZEN_CODE_FILES = (
    "contracts.py",
    "executor.py",
    "synthesis.py",
    "posterior.py",
    "decision.py",
    "replay_gate.py",
)
CONDITIONS = ("joint", "dynamics_only", "random", "action_only")


@dataclass(frozen=True)
class PairedInterval:
    """Deterministic paired bootstrap interval over independent root episodes."""

    n: int
    mean: float
    lower: float
    upper: float
    confidence: float = 0.95


@dataclass(frozen=True)
class StageReplayResult:
    """One condition at one observation checkpoint for one root episode."""

    episode_id: str
    source_game: str
    source_split: str
    condition: str
    observations: int
    held_out_log_likelihood: float
    entropy_reduction: float
    discriminative_action_quality: float
    decision_time_ms: float
    repairs_attempted: int
    repairs_admitted: int
    generated_programs: int
    posterior_programs: int
    correct_family_generated: bool | None = None
    correct_family_top1: bool | None = None
    correct_family_top5: bool | None = None
    correct_family_top16: bool | None = None
    oracle_log_likelihood: float | None = None
    best_generated_log_likelihood: float | None = None
    generation_regret: float | None = None
    selection_regret: float | None = None
    oracle_family_count: int = 0
    generation_failure: bool = False
    selection_failure: bool = False
    forbidden_fields: int = 0
    illegal_actions: int = 0
    execution_errors: int = 0


@dataclass(frozen=True)
class ReplayEpisode:
    """Five independent interventions grouped by a replay-verified V4.3 root."""

    episode_id: str
    source_game: str
    source_split: str
    panels: tuple[CounterfactualPanel, ...]


@dataclass
class _ProgramRun:
    stages: list[StageReplayResult] = field(default_factory=list)
    revealed_keys: list[str] = field(default_factory=list)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _checksum(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(_json_safe(value)).encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical(_json_safe(row)) + "\n")
    os.replace(temporary, path)


def current_code_hashes() -> dict[str, str]:
    """Hash the implementation files whose semantics are frozen for T7."""

    directory = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in FROZEN_CODE_FILES
    }


def load_frozen_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    verify_code: bool = True,
) -> dict[str, Any]:
    """Read and strictly validate the pre-replay freeze."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    checksum = str(payload.get("manifest_checksum", ""))
    unsigned = dict(payload)
    unsigned.pop("manifest_checksum", None)
    if checksum != _checksum(unsigned):
        raise ValueError("SAGE.T7 frozen manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported SAGE.T7 frozen manifest")
    if payload.get("status") != "FROZEN_BEFORE_REPLAY":
        raise ValueError("SAGE.T7 manifest is not frozen")
    if payload.get("observation_checkpoints") != [1, 3, 5]:
        raise ValueError("SAGE.T7 checkpoints must remain [1, 3, 5]")
    if payload.get("firewall", {}).get("holdout_opened") is not False:
        raise ValueError("SAGE.T7 holdout firewall is open")
    if payload.get("likelihood", {}).get("channel_weights") != dict(
        DEFAULT_CHANNEL_WEIGHTS
    ):
        raise ValueError("SAGE.T7 likelihood coefficients drifted")
    if verify_code and payload.get("code_sha256") != current_code_hashes():
        raise ValueError("SAGE.T7 frozen grammar/executor code drifted")
    return payload


def _trace_events(trace: Any) -> tuple[str, ...]:
    effects = trace.effects
    labels = dict(effects.labels)
    events: list[str] = []
    if labels.get("actor_displaced") or labels.get("target_moved"):
        events.append("moved")
    if labels.get("target_created"):
        events.append("created")
    if labels.get("target_removed"):
        events.append("removed")
    if bool(effects.noop):
        events.append("no_effect")
    if bool(effects.level_complete) or (
        int(trace.levels_completed_after) > int(trace.levels_completed_before)
    ):
        events.extend(("progress", "level_complete"))
    if bool(effects.game_over):
        events.append("game_over")
    if not events:
        # V4.3 effect annotations are conservative.  An applicable action with
        # no accepted label remains explicit no-effect evidence.
        events.append("no_effect")
    return tuple(sorted(set(events)))


def _shared_trace_state(pair: BindingPairRecord) -> AbstractState:
    entities = [
        AbstractEntity("e_player", ("object", "player", "movable")),
    ]
    true_facts: set[GroundFact] = set()
    for side, branch in (("left", pair.left), ("right", pair.right)):
        anchor = branch.trace.anchor
        roles = ["object", "target"]
        if str(anchor.target_affordance) == "movable":
            roles.append("movable")
        entity_id = f"e_target_{side}"
        attributes = (
            ("action_family", str(anchor.action_family)),
            ("affordance", str(anchor.target_affordance)),
            ("anchor_kind", str(anchor.kind)),
            ("occupied", str(bool(anchor.occupied)).lower()),
            ("path_status", str(anchor.path_status)),
        )
        center = (
            None
            if anchor.row is None or anchor.col is None
            else (float(anchor.row), float(anchor.col))
        )
        entities.append(
            AbstractEntity(
                entity_id,
                tuple(roles),
                attributes=attributes,
                center=center,
            )
        )
        relation = str(anchor.actor_relation).lower()
        if relation == "adjacent":
            true_facts.add(GroundFact("adjacent", ("e_player", entity_id)))
        elif relation == "contact":
            true_facts.add(GroundFact("contact", ("e_player", entity_id)))
        if bool(anchor.in_bounds):
            true_facts.add(GroundFact("reachable", (entity_id,)))
    return AbstractState(
        entities=tuple(entities),
        true_facts=frozenset(true_facts),
        counters=(("levels_completed", float(pair.left.trace.levels_completed_before)),),
    )


def _fast_transition(
    trace: Any,
    *,
    state_before: AbstractState,
    target_entity_id: str,
) -> ObservedTransition:
    events = _trace_events(trace)
    object_deltas = {
        event: 1.0
        for event in events
        if event
        not in {
            "progress",
            "level_complete",
            "game_over",
        }
    }
    progress = max(
        0.0,
        float(trace.levels_completed_after)
        - float(trace.levels_completed_before),
    )
    after_facts = set(state_before.true_facts)
    if "level_complete" in events:
        after_facts.add(GroundFact("level_complete"))
    if "game_over" in events:
        after_facts.add(GroundFact("game_over"))
    state_after = AbstractState(
        entities=state_before.entities,
        true_facts=frozenset(after_facts),
        false_facts=state_before.false_facts,
        counters=(("levels_completed", float(trace.levels_completed_after)),),
        topology=state_before.topology,
        regime_index=state_before.regime_index,
    )
    known = {"objects", "progress", "terminal"}
    goal_probability = None
    if "level_complete" in events:
        known.add("goal")
        goal_probability = 1.0
    action_data = dict(trace.selected_action_data)
    action_data["entity_id"] = target_entity_id
    observation = PredictionPacket(
        object_deltas=object_deltas,
        progress_mean=progress,
        progress_distribution={f"value:{progress:.6g}": 1.0},
        terminal_probability=float("game_over" in events),
        goal_probability=goal_probability,
        known_channels=frozenset(known),
        state_after=state_after,
    )
    return ObservedTransition(
        state_before=state_before,
        action=ActionCandidate(trace.selected_action_name, action_data),
        state_after=state_after,
        observation=observation,
        events=events,
    )


def fast_panel_from_binding_pair(pair: BindingPairRecord) -> CounterfactualPanel:
    """Compile one V4.3 pair from its frozen semantic annotations.

    This intentionally masks relation/topology channels that V4.3 did not
    annotate.  It is substantially faster than recomputing scene graphs and,
    unlike raw-grid shortcuts, preserves absent/false/unknown semantics.
    """

    if pair.source_split not in {"source_train", "source_validation"}:
        raise ValueError("SAGE.T7 is source-only")
    state = _shared_trace_state(pair)
    arms = (
        _fast_transition(
            pair.left.trace,
            state_before=state,
            target_entity_id="e_target_left",
        ),
        _fast_transition(
            pair.right.trace,
            state_before=state,
            target_entity_id="e_target_right",
        ),
    )
    return CounterfactualPanel(
        panel_id=pair.pair_digest,
        state=state,
        arms=arms,
        source_game=pair.game_id,
    )


def episodes_from_binding_pairs(
    pairs: Sequence[BindingPairRecord],
    *,
    panels_per_episode: int = 5,
) -> tuple[ReplayEpisode, ...]:
    """Group pairs by V4.3 replay root, the paired-bootstrap sampling unit."""

    grouped: dict[str, list[BindingPairRecord]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.root_key].append(pair)
    episodes = []
    required = max(1, int(panels_per_episode))
    for root_key, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: (item.depth, item.path))
        if len(ordered) < required:
            continue
        chosen = ordered[:required]
        episodes.append(
            ReplayEpisode(
                episode_id=root_key,
                source_game=chosen[0].game_id,
                source_split=chosen[0].source_split,
                panels=tuple(fast_panel_from_binding_pair(item) for item in chosen),
            )
        )
    return tuple(episodes)


def _weights(condition: str) -> Mapping[str, float]:
    if condition in {"joint", "random"}:
        return DEFAULT_CHANNEL_WEIGHTS
    if condition == "dynamics_only":
        return {
            **DEFAULT_CHANNEL_WEIGHTS,
            "progress": 0.0,
            "goal": 0.0,
        }
    raise ValueError(f"unsupported program condition: {condition}")


def _programs_for(
    available_actions: Sequence[str],
    transitions: Sequence[ObservedTransition],
    manifest: Mapping[str, Any],
) -> tuple[AssembledProgram, ...]:
    generator = manifest["generator"]
    event_map: dict[str, set[str]] = defaultdict(set)
    for transition in transitions:
        event_map[transition.action.action_name].update(transition.events)
    return _cached_programs_for_events(
        tuple(sorted({str(action).upper() for action in available_actions})),
        tuple(
            (action, tuple(sorted(events)))
            for action, events in sorted(event_map.items())
        ),
        int(generator["maximum_operator_candidates_per_action"]),
        int(generator["maximum_programs"]),
        int(generator["maximum_dynamics_beam"]),
    )


@lru_cache(maxsize=4096)
def _cached_programs_for_events(
    available_actions: tuple[str, ...],
    event_map: tuple[tuple[str, tuple[str, ...]], ...],
    maximum_operator_candidates_per_action: int,
    maximum_programs: int,
    maximum_dynamics_beam: int,
) -> tuple[AssembledProgram, ...]:
    summaries = tuple(
        SimpleNamespace(
            action=SimpleNamespace(action_name=action),
            events=events,
        )
        for action, events in event_map
    )
    proposal = DeterministicFragmentProposer(
        maximum_operator_candidates_per_action=(
            maximum_operator_candidates_per_action
        )
    ).propose(
        available_actions=available_actions,
        transitions=summaries,
    )
    return ProgramAssembler(
        maximum_programs=maximum_programs,
        maximum_dynamics_beam=maximum_dynamics_beam,
    ).assemble(
        proposal.fragments,
        available_actions=available_actions,
    )


def _new_posterior(
    condition: str,
    executor: ProgramExecutor,
    manifest: Mapping[str, Any],
) -> ProgramPosterior:
    config = manifest["posterior"]
    return ProgramPosterior(
        executor=executor,
        maximum_particles=int(config["maximum_particles"]),
        channel_weights=_weights(condition),
        unknown_coverage_penalty=float(config["unknown_coverage_penalty"]),
        repair_ess_threshold=float(config["repair_ess_threshold"]),
        repair_log_likelihood_threshold=float(
            config["repair_log_likelihood_threshold"]
        ),
    )


def _mixture_log_likelihood(
    posterior: ProgramPosterior,
    evidence: Sequence[ObservedTransition],
    executor: ProgramExecutor,
) -> float:
    scores = []
    for arm in evidence:
        terms = []
        for particle in posterior.particles:
            try:
                prediction = executor.step(
                    particle.program,
                    arm.state_before,
                    arm.action,
                )
                likelihood = packet_log_likelihood(
                    prediction,
                    arm.observation,
                    channel_weights=posterior.channel_weights,
                    unknown_coverage_penalty=posterior.unknown_coverage_penalty,
                )
                terms.append(particle.log_weight + likelihood)
            except (TypeError, ValueError, RuntimeError):
                continue
        if terms:
            scores.append(_logsumexp(terms))
    return mean(scores) if scores else float("-inf")


def _individual_program_score(
    program: AssembledProgram,
    evidence: Sequence[ObservedTransition],
    executor: ProgramExecutor,
    weights: Mapping[str, float],
) -> float:
    values = [
        packet_log_likelihood(
            executor.step(program.program, arm.state_before, arm.action),
            arm.observation,
            channel_weights=weights,
        )
        for arm in evidence
    ]
    return mean(values) if values else float("-inf")


def _oracle_families(
    *,
    revealed: Sequence[ObservedTransition],
    hidden: Sequence[ObservedTransition],
    available_actions: Sequence[str],
    manifest: Mapping[str, Any],
    executor: ProgramExecutor,
    generated_programs: Sequence[AssembledProgram] = (),
) -> tuple[frozenset[tuple[str, str]], float]:
    """Return grammar-relative best families, using hidden arms for labels only."""

    oracle_candidates = _programs_for(
        available_actions,
        tuple(revealed) + tuple(hidden),
        manifest,
    )
    programs_by_hash = {
        program.program.canonical_hash: program
        for program in (*oracle_candidates, *generated_programs)
    }
    programs = tuple(programs_by_hash.values())
    if not programs:
        return frozenset(), float("-inf")
    by_family: dict[tuple[str, str], float] = {}
    for program in programs:
        score = _individual_program_score(
            program,
            hidden,
            executor,
            DEFAULT_CHANNEL_WEIGHTS,
        )
        family = program.program.semantic_family
        by_family[family] = max(score, by_family.get(family, float("-inf")))
    best = max(by_family.values())
    tolerance = float(manifest["evaluation"]["oracle_family_tolerance"])
    return (
        frozenset(
            family
            for family, score in by_family.items()
            if best - score <= tolerance
        ),
        best,
    )


def _evidence_entropy_reduction(
    posterior: ProgramPosterior,
    evidence: ObservedTransition,
    executor: ProgramExecutor,
) -> float:
    """Exact one-step entropy reduction without mutating or repairing beliefs."""

    terms = []
    for particle in posterior.particles:
        prediction = executor.step(
            particle.program,
            evidence.state_before,
            evidence.action,
        )
        likelihood = packet_log_likelihood(
            prediction,
            evidence.observation,
            channel_weights=posterior.channel_weights,
            unknown_coverage_penalty=posterior.unknown_coverage_penalty,
        )
        terms.append(particle.log_weight + likelihood)
    normalizer = _logsumexp(terms)
    probabilities = [math.exp(value - normalizer) for value in terms]
    after = -sum(
        probability * math.log(max(probability, 1e-300))
        for probability in probabilities
        if probability > 0.0
    )
    return posterior.entropy - after


def _discriminative_quality(
    posterior: ProgramPosterior,
    panel: CounterfactualPanel,
    selected_key: str,
    executor: ProgramExecutor,
) -> float:
    gains = {
        arm.action.key: _evidence_entropy_reduction(posterior, arm, executor)
        for arm in panel.arms
    }
    best = max(gains.values())
    selected = gains[selected_key]
    scale = max(abs(best), 1e-9)
    return max(0.0, min(1.0, 1.0 - max(0.0, best - selected) / scale))


def _run_program_episode(
    episode: ReplayEpisode,
    *,
    condition: str,
    selection: str,
    manifest: Mapping[str, Any],
    seed: int,
    forced_keys: Sequence[str] = (),
    executor: ProgramExecutor | None = None,
) -> _ProgramRun:
    executor = executor or ProgramExecutor(
        maximum_cache_entries=int(manifest["executor"]["maximum_cache_entries"])
    )
    posterior = _new_posterior(condition, executor, manifest)
    actions = tuple(
        sorted(
            {
                arm.action.action_name
                for panel in episode.panels
                for arm in panel.arms
            }
        )
    )
    initial_programs = _programs_for(actions, (), manifest)
    posterior.seed(initial_programs, initial_state=episode.panels[0].state)
    generated: dict[str, AssembledProgram] = {
        item.program.canonical_hash: item for item in initial_programs
    }
    initial_entropy = posterior.normalized_entropy
    engine = CounterfactualDecisionEngine(
        executor=executor,
        maximum_sequences=int(manifest["decision"]["maximum_sequences"]),
        maximum_particles=int(manifest["decision"]["maximum_particles"]),
        ordinary_horizon=int(manifest["decision"]["ordinary_horizon"]),
    )
    rng = random.Random(seed)
    revealed: list[ObservedTransition] = []
    hidden: list[ObservedTransition] = []
    decision_times = []
    qualities = []
    output = _ProgramRun()
    checkpoints = {int(value) for value in manifest["observation_checkpoints"]}

    for index, panel in enumerate(episode.panels):
        available = {arm.action.key: arm for arm in panel.arms}
        started = time.process_time()
        if index < len(forced_keys):
            selected_key = forced_keys[index]
        elif selection == "random":
            selected_key = rng.choice(tuple(sorted(available)))
        else:
            decision = engine.decide(
                posterior,
                panel.state,
                tuple(arm.action for arm in panel.arms),
            )
            selected_key = "" if decision.action is None else decision.action.key
        elapsed_ms = (time.process_time() - started) * 1000.0
        illegal = int(selected_key not in available)
        if illegal:
            selected_key = min(available)
        quality = _discriminative_quality(
            posterior,
            panel,
            selected_key,
            executor,
        )
        chosen = available[selected_key]
        alternative = next(
            arm for key, arm in available.items() if key != selected_key
        )
        revealed.append(chosen)
        hidden.append(alternative)
        output.revealed_keys.append(selected_key)
        posterior.observe(chosen)
        enriched = _programs_for(actions, revealed, manifest)
        for item in enriched:
            generated.setdefault(item.program.canonical_hash, item)
        posterior.add_programs(enriched)
        # Repair children are genuine generated hypotheses too.  Include them
        # in the coverage side of the decomposition after they have replayed
        # the complete revealed history and survived posterior admission.
        for particle in posterior.particles:
            generated.setdefault(
                particle.program.canonical_hash,
                AssembledProgram(particle.program, particle.log_prior),
            )
        after_snapshot = posterior.snapshot(maximum_programs=0)
        decision_times.append(elapsed_ms)
        qualities.append(quality)

        observations = index + 1
        if observations not in checkpoints:
            continue
        oracle_families, oracle_score = _oracle_families(
            revealed=revealed,
            hidden=hidden,
            available_actions=actions,
            manifest=manifest,
            executor=executor,
            generated_programs=tuple(generated.values()),
        )
        generated_families = {
            item.program.semantic_family for item in generated.values()
        }
        top = posterior.top(max(manifest["top_k"]))

        def present(
            limit: int,
            *,
            candidates: Sequence[Any] = top,
            families: frozenset[tuple[str, str]] = oracle_families,
        ) -> bool:
            return any(
                particle.program.semantic_family in families
                for particle in candidates[:limit]
            )

        covered = bool(generated_families & oracle_families)
        top1 = present(1)
        top5 = present(5)
        top16 = present(16)
        best_generated_score = max(
            (
                _individual_program_score(
                    program,
                    hidden,
                    executor,
                    DEFAULT_CHANNEL_WEIGHTS,
                )
                for program in generated.values()
            ),
            default=float("-inf"),
        )
        posterior_score = _mixture_log_likelihood(
            posterior,
            hidden,
            executor,
        )
        output.stages.append(
            StageReplayResult(
                episode_id=episode.episode_id,
                source_game=episode.source_game,
                source_split=episode.source_split,
                condition=condition,
                observations=observations,
                held_out_log_likelihood=posterior_score,
                entropy_reduction=initial_entropy - posterior.normalized_entropy,
                discriminative_action_quality=mean(qualities),
                decision_time_ms=mean(decision_times),
                repairs_attempted=int(after_snapshot["repairs_attempted"]),
                repairs_admitted=int(after_snapshot["repairs_admitted"]),
                generated_programs=len(generated),
                posterior_programs=len(posterior.particles),
                correct_family_generated=covered,
                correct_family_top1=top1,
                correct_family_top5=top5,
                correct_family_top16=top16,
                oracle_log_likelihood=oracle_score,
                best_generated_log_likelihood=best_generated_score,
                generation_regret=max(0.0, oracle_score - best_generated_score),
                selection_regret=max(
                    0.0,
                    best_generated_score - posterior_score,
                ),
                oracle_family_count=len(oracle_families),
                generation_failure=not covered,
                selection_failure=covered and not top16,
                forbidden_fields=count_forbidden_program_fields(
                    tuple(generated.values())
                ),
                illegal_actions=illegal,
            )
        )
    return output


def _mean_packet(packets: Sequence[PredictionPacket]) -> PredictionPacket:
    if not packets:
        return PredictionPacket()
    known = set(packets[0].known_channels)
    for packet in packets[1:]:
        known.intersection_update(packet.known_channels)

    def events(attribute: str) -> dict[str, float]:
        keys = {key for packet in packets for key in dict(getattr(packet, attribute))}
        return {
            key: mean(
                float(dict(getattr(packet, attribute)).get(key, 0.05))
                for packet in packets
            )
            for key in keys
        }

    def scalar(attribute: str) -> float | None:
        values = [
            float(value)
            for packet in packets
            for value in (getattr(packet, attribute),)
            if value is not None
        ]
        return mean(values) if values else None

    return PredictionPacket(
        object_deltas=events("object_deltas"),
        relation_deltas=events("relation_deltas"),
        topology_deltas=events("topology_deltas"),
        progress_mean=scalar("progress_mean"),
        terminal_probability=scalar("terminal_probability"),
        goal_probability=scalar("goal_probability"),
        known_channels=frozenset(known),
    )


def _run_action_only_episode(
    episode: ReplayEpisode,
    *,
    forced_keys: Sequence[str],
    manifest: Mapping[str, Any],
) -> list[StageReplayResult]:
    learned: dict[str, list[PredictionPacket]] = defaultdict(list)
    hidden: list[ObservedTransition] = []
    checkpoints = {int(value) for value in manifest["observation_checkpoints"]}
    output = []
    for index, (panel, selected_key) in enumerate(zip(episode.panels, forced_keys)):
        available = {arm.action.key: arm for arm in panel.arms}
        chosen = available[selected_key]
        hidden.append(
            next(arm for key, arm in available.items() if key != selected_key)
        )
        learned[chosen.action.action_name].append(chosen.observation)
        observations = index + 1
        if observations not in checkpoints:
            continue
        scores = [
            packet_log_likelihood(
                _mean_packet(learned.get(arm.action.action_name, ())),
                arm.observation,
            )
            for arm in hidden
        ]
        output.append(
            StageReplayResult(
                episode_id=episode.episode_id,
                source_game=episode.source_game,
                source_split=episode.source_split,
                condition="action_only",
                observations=observations,
                held_out_log_likelihood=mean(scores),
                entropy_reduction=0.0,
                discriminative_action_quality=0.0,
                decision_time_ms=0.0,
                repairs_attempted=0,
                repairs_admitted=0,
                generated_programs=0,
                posterior_programs=0,
            )
        )
    return output


def run_replay_episodes(
    episodes: Sequence[ReplayEpisode],
    *,
    manifest: Mapping[str, Any],
) -> tuple[StageReplayResult, ...]:
    """Run all four frozen conditions without counterfactual leakage."""

    rows: list[StageReplayResult] = []
    base_seed = int(manifest["evaluation"]["random_seed"])
    executor = ProgramExecutor(
        maximum_cache_entries=int(manifest["executor"]["maximum_cache_entries"])
    )
    for episode_index, episode in enumerate(episodes):
        seed = base_seed + episode_index
        joint = _run_program_episode(
            episode,
            condition="joint",
            selection="information",
            manifest=manifest,
            seed=seed,
            executor=executor,
        )
        dynamics = _run_program_episode(
            episode,
            condition="dynamics_only",
            selection="forced",
            forced_keys=joint.revealed_keys,
            manifest=manifest,
            seed=seed,
            executor=executor,
        )
        random_run = _run_program_episode(
            episode,
            condition="random",
            selection="random",
            manifest=manifest,
            seed=seed,
            executor=executor,
        )
        rows.extend(joint.stages)
        rows.extend(dynamics.stages)
        rows.extend(random_run.stages)
        rows.extend(
            _run_action_only_episode(
                episode,
                forced_keys=joint.revealed_keys,
                manifest=manifest,
            )
        )
    return tuple(rows)


def _run_game_shard_batch(
    payload: tuple[
        str,
        str,
        int,
        int | None,
        Mapping[str, Any],
    ],
) -> tuple[StageReplayResult, ...]:
    shard_dir, game, panels_per_episode, maximum_episodes, manifest = payload
    pairs = load_pairs(shard_dir, (game,))
    episodes = episodes_from_binding_pairs(
        pairs,
        panels_per_episode=panels_per_episode,
    )
    if maximum_episodes is not None:
        episodes = episodes[: max(0, int(maximum_episodes))]
    return run_replay_episodes(episodes, manifest=manifest)


def paired_bootstrap_interval(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> PairedInterval:
    """Bootstrap the mean of exact episode-level paired differences."""

    keys = tuple(sorted(set(left) & set(right)))
    differences = [float(left[key]) - float(right[key]) for key in keys]
    if not differences:
        return PairedInterval(0, float("nan"), float("nan"), float("nan"), confidence)
    rng = random.Random(seed)
    estimates = []
    for _ in range(max(1, int(samples))):
        draw = [rng.choice(differences) for _ in differences]
        estimates.append(mean(draw))
    estimates.sort()
    alpha = (1.0 - float(confidence)) / 2.0
    return PairedInterval(
        n=len(differences),
        mean=mean(differences),
        lower=_quantile(estimates, alpha),
        upper=_quantile(estimates, 1.0 - alpha),
        confidence=float(confidence),
    )


def _condition_rows(
    rows: Sequence[StageReplayResult],
    *,
    condition: str,
    checkpoint: int,
    game: str | None = None,
) -> dict[str, StageReplayResult]:
    return {
        row.episode_id: row
        for row in rows
        if row.condition == condition
        and row.observations == checkpoint
        and (game is None or row.source_game == game)
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    position = max(0.0, min(1.0, probability)) * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)


def _group_mean(
    rows: Sequence[StageReplayResult],
    attribute: str,
) -> float:
    values = [
        float(getattr(row, attribute))
        for row in rows
        if math.isfinite(float(getattr(row, attribute)))
    ]
    return mean(values) if values else float("nan")


def build_replay_report(
    rows: Sequence[StageReplayResult],
    *,
    manifest: Mapping[str, Any],
    expected_games: Sequence[str],
) -> dict[str, Any]:
    """Aggregate per-game metrics, paired intervals, gates, and root cause."""

    expected = tuple(str(game) for game in expected_games)
    games = tuple(sorted({row.source_game for row in rows}))
    checkpoints = tuple(int(value) for value in manifest["observation_checkpoints"])
    per_game: dict[str, Any] = {}
    for game in games:
        per_game[game] = {}
        for checkpoint in checkpoints:
            per_game[game][str(checkpoint)] = {}
            for condition in CONDITIONS:
                selected = [
                    row
                    for row in rows
                    if row.source_game == game
                    and row.observations == checkpoint
                    and row.condition == condition
                ]
                if not selected:
                    continue
                per_game[game][str(checkpoint)][condition] = {
                    "episodes": len(selected),
                    "held_out_log_likelihood": _group_mean(
                        selected, "held_out_log_likelihood"
                    ),
                    "entropy_reduction": _group_mean(
                        selected, "entropy_reduction"
                    ),
                    "discriminative_action_quality": _group_mean(
                        selected, "discriminative_action_quality"
                    ),
                    "decision_time_ms": _group_mean(selected, "decision_time_ms"),
                    "repair_trigger_rate": mean(
                        row.repairs_attempted > 0 for row in selected
                    ),
                    "repair_parent_attempts_per_observation": (
                        sum(row.repairs_attempted for row in selected)
                        / max(1, checkpoint * len(selected))
                    ),
                    "repair_children_per_parent_attempt": (
                        sum(row.repairs_admitted for row in selected)
                        / max(1, sum(row.repairs_attempted for row in selected))
                    ),
                    "correct_family_generated_rate": _optional_bool_mean(
                        row.correct_family_generated for row in selected
                    ),
                    "correct_family_top1_rate": _optional_bool_mean(
                        row.correct_family_top1 for row in selected
                    ),
                    "correct_family_top5_rate": _optional_bool_mean(
                        row.correct_family_top5 for row in selected
                    ),
                    "correct_family_top16_rate": _optional_bool_mean(
                        row.correct_family_top16 for row in selected
                    ),
                    "oracle_log_likelihood": _optional_float_mean(
                        row.oracle_log_likelihood for row in selected
                    ),
                    "best_generated_log_likelihood": _optional_float_mean(
                        row.best_generated_log_likelihood for row in selected
                    ),
                    "generation_regret": _optional_float_mean(
                        row.generation_regret for row in selected
                    ),
                    "selection_regret": _optional_float_mean(
                        row.selection_regret for row in selected
                    ),
                }

    comparisons: dict[str, Any] = {}
    samples = int(manifest["evaluation"]["bootstrap_samples"])
    seed = int(manifest["evaluation"]["random_seed"])
    confidence = float(manifest["evaluation"]["paired_confidence"])
    for checkpoint in checkpoints:
        selected = [row for row in rows if row.observations == checkpoint]
        by_condition = {
            condition: {
                row.episode_id: row
                for row in selected
                if row.condition == condition
            }
            for condition in CONDITIONS
        }
        comparisons[str(checkpoint)] = {
            "joint_log_likelihood_minus_action_only": asdict(
                paired_bootstrap_interval(
                    {
                        key: row.held_out_log_likelihood
                        for key, row in by_condition["joint"].items()
                    },
                    {
                        key: row.held_out_log_likelihood
                        for key, row in by_condition["action_only"].items()
                    },
                    samples=samples,
                    seed=seed + checkpoint * 11,
                    confidence=confidence,
                )
            ),
            "joint_log_likelihood_minus_dynamics_only": asdict(
                paired_bootstrap_interval(
                    {
                        key: row.held_out_log_likelihood
                        for key, row in by_condition["joint"].items()
                    },
                    {
                        key: row.held_out_log_likelihood
                        for key, row in by_condition["dynamics_only"].items()
                    },
                    samples=samples,
                    seed=seed + checkpoint * 13,
                    confidence=confidence,
                )
            ),
            "joint_entropy_reduction_minus_random": asdict(
                paired_bootstrap_interval(
                    {
                        key: row.entropy_reduction
                        for key, row in by_condition["joint"].items()
                    },
                    {
                        key: row.entropy_reduction
                        for key, row in by_condition["random"].items()
                    },
                    samples=samples,
                    seed=seed + checkpoint * 17,
                    confidence=confidence,
                )
            ),
            "joint_action_quality_minus_random": asdict(
                paired_bootstrap_interval(
                    {
                        key: row.discriminative_action_quality
                        for key, row in by_condition["joint"].items()
                    },
                    {
                        key: row.discriminative_action_quality
                        for key, row in by_condition["random"].items()
                    },
                    samples=samples,
                    seed=seed + checkpoint * 19,
                    confidence=confidence,
                )
            ),
        }

    final_checkpoint = max(checkpoints)
    final_joint = [
        row
        for row in rows
        if row.condition == "joint" and row.observations == final_checkpoint
    ]
    generation_failures = sum(row.generation_failure for row in final_joint)
    selection_failures = sum(row.selection_failure for row in final_joint)
    successes = sum(bool(row.correct_family_top16) for row in final_joint)
    generation_regret = _optional_float_mean(
        row.generation_regret for row in final_joint
    )
    selection_regret = _optional_float_mean(
        row.selection_regret for row in final_joint
    )
    generation_regret_value = float(generation_regret or 0.0)
    selection_regret_value = float(selection_regret or 0.0)
    if generation_regret_value > selection_regret_value + 1e-9:
        bottleneck = "generator_coverage"
    elif selection_regret_value > generation_regret_value + 1e-9:
        bottleneck = "posterior_selection"
    elif generation_regret_value > 1e-9 or selection_regret_value > 1e-9:
        bottleneck = "mixed"
    elif generation_failures > selection_failures:
        bottleneck = "generator_coverage"
    elif selection_failures > generation_failures:
        bottleneck = "posterior_selection"
    elif generation_failures or selection_failures:
        bottleneck = "mixed"
    else:
        bottleneck = "neither_detected_within_frozen_grammar"

    final_comparisons = comparisons[str(final_checkpoint)]
    per_game_primary_deltas = {}
    for game in games:
        joint_rows = _condition_rows(
            rows,
            condition="joint",
            checkpoint=final_checkpoint,
            game=game,
        )
        action_rows = _condition_rows(
            rows,
            condition="action_only",
            checkpoint=final_checkpoint,
            game=game,
        )
        dynamics_rows = _condition_rows(
            rows,
            condition="dynamics_only",
            checkpoint=final_checkpoint,
            game=game,
        )
        keys = sorted(set(joint_rows) & set(action_rows) & set(dynamics_rows))
        per_game_primary_deltas[game] = (
            mean(
                joint_rows[key].held_out_log_likelihood
                - max(
                    action_rows[key].held_out_log_likelihood,
                    dynamics_rows[key].held_out_log_likelihood,
                )
                for key in keys
            )
            if keys
            else float("nan")
        )
    non_negative_games = sum(
        math.isfinite(value) and value >= 0.0
        for value in per_game_primary_deltas.values()
    )
    interval_threshold = float(
        manifest["gates"]["paired_interval_lower_must_exceed"]
    )
    safety = {
        "zero_illegal_actions": sum(row.illegal_actions for row in rows) == 0,
        "zero_execution_errors": sum(row.execution_errors for row in rows) == 0,
        "zero_forbidden_fields": sum(row.forbidden_fields for row in rows) == 0,
        "all_expected_games_present": set(games) == set(expected),
        "all_metrics_finite": all(
            math.isfinite(row.held_out_log_likelihood) for row in rows
        ),
        "two_of_three_games_non_negative": (
            len(expected) != 3 or non_negative_games >= 2
        ),
    }
    predictive = {
        "joint_beats_action_only": (
            final_comparisons["joint_log_likelihood_minus_action_only"]["lower"]
            > interval_threshold
        ),
        "joint_beats_dynamics_only": (
            final_comparisons["joint_log_likelihood_minus_dynamics_only"]["lower"]
            > interval_threshold
        ),
        "information_beats_random": (
            final_comparisons["joint_entropy_reduction_minus_random"]["lower"]
            > interval_threshold
        ),
        "discriminative_action_beats_random": (
            final_comparisons["joint_action_quality_minus_random"]["lower"]
            > interval_threshold
        ),
    }
    checks = {**safety, **predictive}
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "PASS" if all(checks.values()) else "FAIL_CLOSED",
        "manifest_checksum": manifest["manifest_checksum"],
        "source_split": (rows[0].source_split if rows else ""),
        "games": list(games),
        "episodes": len({row.episode_id for row in rows}),
        "stage_rows": len(rows),
        "per_game": per_game,
        "paired_intervals": comparisons,
        "per_game_primary_delta_at_5": per_game_primary_deltas,
        "non_negative_games_at_5": non_negative_games,
        "gate_checks": checks,
        "diagnosis": {
            "oracle_definition": (
                "best held-out predictive semantic family within the frozen grammar"
            ),
            "generation_failures_at_5": generation_failures,
            "selection_failures_at_5": selection_failures,
            "top16_successes_at_5": successes,
            "episodes_at_5": len(final_joint),
            "mean_generation_regret_at_5": generation_regret,
            "mean_selection_regret_at_5": selection_regret,
            "primary_bottleneck": bottleneck,
            "limitation": (
                "The oracle family is grammar-relative; failure of the oracle "
                "itself indicates representation/executor misspecification."
            ),
        },
        "firewall": {
            "source_only": True,
            "holdout_opened": False,
            "ar25_opened": False,
        },
    }
    unsigned = dict(payload)
    payload["report_checksum"] = _checksum(unsigned)
    return payload


def _optional_bool_mean(values: Iterable[bool | None]) -> float | None:
    items = [bool(value) for value in values if value is not None]
    return mean(items) if items else None


def _optional_float_mean(values: Iterable[float | None]) -> float | None:
    items = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return mean(items) if items else None


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    maximum = max(values)
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def run_split(
    *,
    split: str,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    v43_dir: str | Path = DEFAULT_V43_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    maximum_episodes_per_game: int | None = None,
    verify_code: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    """Execute one frozen source split and persist raw and aggregate reports."""

    manifest = load_frozen_manifest(manifest_path, verify_code=verify_code)
    if split not in {"source_train", "source_validation"}:
        raise ValueError("SAGE.T7 split must be source_train/source_validation")
    destination = Path(output_dir)
    if split == "source_validation":
        train_path = destination / "source_train_report.json"
        if not train_path.exists():
            payload = {
                "format_version": FORMAT_VERSION,
                "status": "BLOCKED_BY_SOURCE_TRAIN_GATE",
                "manifest_checksum": manifest["manifest_checksum"],
                "source_split": split,
                "reason": "missing_source_train_report",
                "holdout_opened": False,
            }
            payload["report_checksum"] = _checksum(payload)
            _write_json(destination / f"{split}_report.json", payload)
            return payload
        train = json.loads(train_path.read_text(encoding="utf-8"))
        signed = dict(train)
        checksum = str(signed.pop("report_checksum", ""))
        if checksum != _checksum(signed) or train.get("status") != "PASS":
            payload = {
                "format_version": FORMAT_VERSION,
                "status": "BLOCKED_BY_SOURCE_TRAIN_GATE",
                "manifest_checksum": manifest["manifest_checksum"],
                "source_split": split,
                "reason": "source_train_gate_not_passed",
                "source_train_report_checksum": checksum,
                "holdout_opened": False,
            }
            payload["report_checksum"] = _checksum(payload)
            _write_json(destination / f"{split}_report.json", payload)
            return payload
    games = tuple(manifest[f"{split}_games"])
    root = Path(v43_dir)
    shard_dir = root / f"{split}_shards"
    missing = [
        str(shard_dir / f"{game}.jsonl")
        for game in games
        if not (shard_dir / f"{game}.jsonl").exists()
    ]
    if missing:
        payload = {
            "format_version": FORMAT_VERSION,
            "status": f"BLOCKED_MISSING_{split.upper()}_SHARDS",
            "manifest_checksum": manifest["manifest_checksum"],
            "source_split": split,
            "missing_shards": missing,
            "holdout_opened": False,
        }
        payload["report_checksum"] = _checksum(payload)
        _write_json(Path(output_dir) / f"{split}_report.json", payload)
        return payload
    worker_count = max(1, int(workers))
    panels_per_episode = int(manifest["episode"]["panels_per_episode"])
    if worker_count == 1:
        pairs = load_pairs(shard_dir, games)
        episodes = list(
            episodes_from_binding_pairs(
                pairs,
                panels_per_episode=panels_per_episode,
            )
        )
        if maximum_episodes_per_game is not None:
            retained = []
            counts: dict[str, int] = defaultdict(int)
            for episode in episodes:
                if counts[episode.source_game] >= int(maximum_episodes_per_game):
                    continue
                retained.append(episode)
                counts[episode.source_game] += 1
            episodes = retained
        rows = run_replay_episodes(episodes, manifest=manifest)
    else:
        batches = [
            (
                str(shard_dir),
                game,
                panels_per_episode,
                maximum_episodes_per_game,
                manifest,
            )
            for game in games
        ]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count
        ) as pool:
            rows = tuple(
                row
                for batch_rows in pool.map(_run_game_shard_batch, batches)
                for row in batch_rows
            )
    report = build_replay_report(rows, manifest=manifest, expected_games=games)
    report["execution_workers"] = worker_count
    unsigned = dict(report)
    unsigned.pop("report_checksum", None)
    report["report_checksum"] = _checksum(unsigned)
    _write_jsonl(
        destination / f"{split}_rows.jsonl",
        (asdict(row) for row in rows),
    )
    _write_json(destination / f"{split}_report.json", report)
    return report


def load_stage_rows(path: str | Path) -> tuple[StageReplayResult, ...]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(StageReplayResult(**json.loads(line)))
    return tuple(rows)


def rebuild_split_report(
    *,
    split: str,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    verify_code: bool = True,
) -> dict[str, Any]:
    """Rebuild aggregates from immutable raw rows without rerunning inference."""

    if split not in {"source_train", "source_validation"}:
        raise ValueError("SAGE.T7 split must be source_train/source_validation")
    manifest = load_frozen_manifest(manifest_path, verify_code=verify_code)
    destination = Path(output_dir)
    rows = load_stage_rows(destination / f"{split}_rows.jsonl")
    report = build_replay_report(
        rows,
        manifest=manifest,
        expected_games=tuple(manifest[f"{split}_games"]),
    )
    previous_path = destination / f"{split}_report.json"
    workers = 1
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        workers = int(previous.get("execution_workers", 1))
    report["execution_workers"] = workers
    unsigned = dict(report)
    unsigned.pop("report_checksum", None)
    report["report_checksum"] = _checksum(unsigned)
    _write_json(previous_path, report)
    return report


def _write_gate_summary(
    *,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    validation_authorized = train.get("status") == "PASS"
    passed = validation.get("status") == "PASS"
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "PASS" if passed else "FAIL_CLOSED",
        "source_train": {
            "status": train.get("status"),
            "report_checksum": train.get("report_checksum"),
        },
        "source_validation": {
            "authorized": validation_authorized,
            "status": validation.get("status"),
            "report_checksum": validation.get("report_checksum"),
        },
        "holdout_opened": False,
        "active_authority_authorized": passed,
    }
    payload["gate_checksum"] = _checksum(payload)
    _write_json(Path(output_dir) / "gate_report.json", payload)
    return payload


def run_all(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    v43_dir: str | Path = DEFAULT_V43_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    maximum_episodes_per_game: int | None = None,
    verify_code: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    """Run source-train, then open source-validation only after a train pass."""

    common = {
        "manifest_path": manifest_path,
        "v43_dir": v43_dir,
        "output_dir": output_dir,
        "maximum_episodes_per_game": maximum_episodes_per_game,
        "verify_code": verify_code,
        "workers": workers,
    }
    train = run_split(split="source_train", **common)
    validation = run_split(split="source_validation", **common)
    return _write_gate_summary(
        train=train,
        validation=validation,
        output_dir=output_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "source-train",
            "source-validation",
            "rebuild-source-train",
            "rebuild-source-validation",
            "all",
        ),
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--v43-dir", default=str(DEFAULT_V43_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--maximum-episodes-per-game", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    common = {
        "manifest_path": args.manifest,
        "v43_dir": args.v43_dir,
        "output_dir": args.output_dir,
        "maximum_episodes_per_game": args.maximum_episodes_per_game,
        "workers": args.workers,
    }
    if args.command == "all":
        result = run_all(**common)
    elif args.command.startswith("rebuild-"):
        split = args.command.removeprefix("rebuild-").replace("-", "_")
        result = rebuild_split_report(
            split=split,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
        )
    else:
        result = run_split(split=args.command.replace("-", "_"), **common)
    print(
        json.dumps(
            _json_safe(
                {
                    "status": result.get("status"),
                    "diagnosis": result.get("diagnosis"),
                    "gate_checks": result.get("gate_checks"),
                    "source_train": result.get("source_train"),
                    "source_validation": result.get("source_validation"),
                    "output_dir": str(args.output_dir),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_V43_DIR",
    "FORMAT_VERSION",
    "PairedInterval",
    "ReplayEpisode",
    "StageReplayResult",
    "build_replay_report",
    "current_code_hashes",
    "episodes_from_binding_pairs",
    "fast_panel_from_binding_pair",
    "load_frozen_manifest",
    "main",
    "paired_bootstrap_interval",
    "rebuild_split_report",
    "run_all",
    "run_replay_episodes",
    "run_split",
]
