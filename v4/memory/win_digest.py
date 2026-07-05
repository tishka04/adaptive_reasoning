"""Compact per-game win digest for V4.

After an agent finishes a game (WIN or timeout), this module extracts a
very compact (~0.5-2 KB) summary of what was USEFUL versus USELESS. The
digest is stored in cross-game memory keyed by game_id so that future
runs can bias priors without persisting heavy per-frame traces.

Attribution policy (matches user choice "both"):
  - LEVEL-TRANSITION primary:
      operators/primitives that appear in `memory.game.all_traces`
      (compressed per-level winning trajectories) are marked useful.
  - WIN-TRAJECTORY confirmation:
      if the game was won, operators/primitives appearing across all
      compressed level traces get an additional "confirmed" flag.
  - Everything tried (support > 0) but never crossing a level AND never
      appearing in any compressed trace is marked useless.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# Thresholds for confidence gating (kept small & explicit for auditability).
USEFUL_CONF_MIN = 0.45
USEFUL_SUPPORT_MIN = 2
USELESS_SUPPORT_MIN = 3
USELESS_CONF_MAX = 0.25
USEFUL_RITUAL_SUCCESS_MIN = 0.45
USEFUL_MOTIF_TERMINAL_MIN = 0.25
USEFUL_MOTIF_SURVIVAL_MIN = 0.20


@dataclass
class GameWinDigest:
    """Very compact per-game record. Target size: <2 KB pickled."""

    game_id: str
    won: bool
    levels_completed: int
    max_level_reached: int
    total_actions: int
    solution_length: int                              # sum of compressed level traces
    per_level_action_counts: list[int] = field(default_factory=list)
    ontology: str = "unknown"
    ontology_confidence: float = 0.0

    # Primitive-level (ACTION1..7, RESET, plus click coords coarsely binned)
    useful_primitives: list[str] = field(default_factory=list)
    useless_primitives: list[str] = field(default_factory=list)

    # Operator-level
    useful_operator_ids: list[str] = field(default_factory=list)
    useful_operator_kinds: list[str] = field(default_factory=list)
    useless_operator_kinds: list[str] = field(default_factory=list)
    confirmed_operator_ids: list[str] = field(default_factory=list)  # in WIN trace

    # Rituals and motifs
    useful_ritual_ids: list[str] = field(default_factory=list)
    useful_motif_ids: list[str] = field(default_factory=list)

    # Headline stats for reporting (small floats, rounded)
    knowledge_level: float = 0.0
    pred_accuracy: float = 0.0
    control_success: float = 0.0
    lp: float = 0.0
    sp: float = 0.0
    tp: float = 0.0

    # Metadata
    stop_reason: str = "unknown"   # "win" | "stagnation" | "time_ceiling"
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameWinDigest":
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {k: v for k, v in data.items() if k in allowed}
        return cls(**clean)


def _primitives_in_traces(all_traces: list[list[Any]]) -> set[str]:
    seen: set[str] = set()
    for trace in all_traces:
        for prim in trace:
            name = getattr(prim, "name", None) or str(prim)
            seen.add(name)
    return seen


def build_digest(
    *,
    game_id: str,
    memory: Any,
    won: bool,
    stop_reason: str,
    elapsed_seconds: float,
) -> GameWinDigest:
    """Build a compact digest from a finished V4 game.

    `memory` is the V4 `MemoryStack` (so we read `memory.game.*`).
    """
    game = memory.game
    all_traces: list[list[Any]] = list(getattr(game, "all_traces", []))
    win_primitives = _primitives_in_traces(all_traces)

    # --- primitives --------------------------------------------------------
    tried_primitives: set[str] = set(getattr(game.profiler, "stats", {}).keys())
    useful_primitives = sorted(win_primitives)
    useless_primitives = sorted(tried_primitives - win_primitives - {"RESET"})

    # --- operators ---------------------------------------------------------
    useful_op_ids: list[str] = []
    useful_op_kinds: set[str] = set()
    confirmed_op_ids: list[str] = []
    useless_op_kinds: set[str] = set()

    for op_id, op in game.inducer.operators.items():
        conf = float(getattr(op, "confidence", 0.0))
        support = int(getattr(op, "support", 0))
        contradictions = int(getattr(op, "contradictions", 0))
        primitive_name = getattr(op, "primitive_action", None)

        is_useful = (
            conf >= USEFUL_CONF_MIN
            and support >= USEFUL_SUPPORT_MIN
            and support >= contradictions
        )
        is_useless = (
            support >= USELESS_SUPPORT_MIN
            and conf <= USELESS_CONF_MAX
        ) or (
            support >= USELESS_SUPPORT_MIN
            and contradictions >= 2 * max(support, 1)
        )

        # WIN-trajectory confirmation: operator's primitive appeared in traces
        confirmed = primitive_name in win_primitives if primitive_name else False

        if is_useful:
            useful_op_ids.append(op_id)
            useful_op_kinds.add(str(op.kind))
            if confirmed:
                confirmed_op_ids.append(op_id)
        elif is_useless:
            useless_op_kinds.add(str(op.kind))

    # --- rituals -----------------------------------------------------------
    useful_ritual_ids = [
        r.ritual_id
        for r in game.rituals.values()
        if float(getattr(r, "success_rate", 0.0)) >= USEFUL_RITUAL_SUCCESS_MIN
    ]

    # --- motifs ------------------------------------------------------------
    useful_motif_ids = [
        m.motif_id
        for m in game.motifs.values()
        if (
            float(getattr(m, "terminal_association", 0.0)) >= USEFUL_MOTIF_TERMINAL_MIN
            or float(getattr(m, "survival_score", 0.0)) >= USEFUL_MOTIF_SURVIVAL_MIN
        )
    ]

    # --- ontology ----------------------------------------------------------
    ontology = "unknown"
    ontology_conf = 0.0
    if game.current_ontologies:
        top = game.current_ontologies[0]
        ontology = str(getattr(top, "kind", "unknown"))
        ontology_conf = float(getattr(top, "confidence", 0.0))

    # --- progress scores ---------------------------------------------------
    try:
        lp, sp, tp = game.progress.scores()
    except Exception:
        lp, sp, tp = 0.0, 0.0, 0.0

    per_level_counts = [len(trace) for trace in all_traces]
    solution_length = sum(per_level_counts)

    # Budget the output: cap list sizes to keep <2 KB.
    return GameWinDigest(
        game_id=game_id,
        won=won,
        levels_completed=int(getattr(game, "total_levels_completed", 0)),
        max_level_reached=int(getattr(game, "max_level_reached", 0)),
        total_actions=int(getattr(game, "total_actions", 0)),
        solution_length=solution_length,
        per_level_action_counts=per_level_counts[:20],
        ontology=ontology,
        ontology_confidence=round(ontology_conf, 3),
        useful_primitives=useful_primitives[:10],
        useless_primitives=useless_primitives[:10],
        useful_operator_ids=useful_op_ids[:12],
        useful_operator_kinds=sorted(useful_op_kinds)[:8],
        useless_operator_kinds=sorted(useless_op_kinds)[:8],
        confirmed_operator_ids=confirmed_op_ids[:12],
        useful_ritual_ids=useful_ritual_ids[:4],
        useful_motif_ids=useful_motif_ids[:6],
        knowledge_level=round(float(game.knowledge_level()), 3),
        pred_accuracy=round(float(game.inducer.operator_predictive_accuracy()), 3),
        control_success=round(float(game.inducer.operator_control_success()), 3),
        lp=round(float(lp), 3),
        sp=round(float(sp), 3),
        tp=round(float(tp), 3),
        stop_reason=stop_reason,
        elapsed_seconds=round(float(elapsed_seconds), 1),
    )


def merge_into_cross_game(cross_game: Any, digest: GameWinDigest) -> None:
    """Store (or update) the digest inside CrossGameMemoryV4.

    Policy:
      - Overwrite if new digest has more levels completed, OR
        same levels but a shorter solution_length, OR
        previous digest didn't win but this one does.
      - Otherwise keep the best previous digest.
    """
    if not hasattr(cross_game, "game_digests"):
        cross_game.game_digests = {}

    existing = cross_game.game_digests.get(digest.game_id)
    if existing is not None:
        prev = GameWinDigest.from_dict(existing)
        better = False
        if digest.won and not prev.won:
            better = True
        elif digest.levels_completed > prev.levels_completed:
            better = True
        elif (
            digest.levels_completed == prev.levels_completed
            and digest.won == prev.won
            and 0 < digest.solution_length < prev.solution_length
        ):
            better = True
        if not better:
            return
    cross_game.game_digests[digest.game_id] = digest.to_dict()
