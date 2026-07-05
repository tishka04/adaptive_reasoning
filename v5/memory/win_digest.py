"""Compact per-game digest for V5.

Shared with V4's digest concept but adapted to V3-style memory. Target
size: < 2 KB pickled per game.

Attribution policy:
  - LEVEL-TRANSITION primary:
      operators / primitives that appear in winning level traces are
      marked useful.
  - WIN-TRAJECTORY confirmation:
      when the game is WON, operators confirmed by appearing in
      compressed level traces are flagged.
  - Useless: operators tried (support ≥ 3) but consistently failing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, List


# Thresholds — kept small and explicit for auditability.
USEFUL_CONF_MIN = 0.45
USEFUL_SUPPORT_MIN = 2
USELESS_SUPPORT_MIN = 3
USELESS_CONF_MAX = 0.25
USEFUL_RITUAL_SUCCESS_MIN = 0.45


@dataclass
class GameWinDigest:
    """Very compact per-game record. Target size: <2 KB pickled."""

    game_id: str
    won: bool
    levels_completed: int
    max_level_reached: int
    total_actions: int
    solution_length: int
    per_level_action_counts: List[int] = field(default_factory=list)
    ontology: str = "unknown"
    ontology_confidence: float = 0.0

    useful_primitives: List[str] = field(default_factory=list)
    useless_primitives: List[str] = field(default_factory=list)

    useful_operator_ids: List[str] = field(default_factory=list)
    useful_operator_kinds: List[str] = field(default_factory=list)
    useless_operator_kinds: List[str] = field(default_factory=list)
    confirmed_operator_ids: List[str] = field(default_factory=list)

    useful_ritual_ids: List[str] = field(default_factory=list)

    knowledge_level: float = 0.0
    pred_accuracy: float = 0.0
    control_success: float = 0.0
    lp: float = 0.0
    sp: float = 0.0
    tp: float = 0.0

    stop_reason: str = "unknown"
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameWinDigest":
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {k: v for k, v in data.items() if k in allowed}
        return cls(**clean)


def _primitives_in(traces: Iterable[List[Any]]) -> set[str]:
    """Collect unique primitive names across all traces."""
    seen: set[str] = set()
    for trace in traces:
        for prim in trace:
            name = getattr(prim, "name", None) or str(prim)
            seen.add(name)
    return seen


def build_digest(
    *,
    game_id: str,
    won: bool,
    stop_reason: str,
    elapsed_seconds: float,
    total_actions: int,
    levels_completed: int,
    max_level_reached: int,
    operators: dict[str, Any],       # memory.inducer.operators
    profiler_stats_keys: Iterable[str],
    all_traces: Iterable[List[Any]],  # list of compressed level traces
    rituals: Iterable[Any],
    top_ontology: str,
    top_ontology_confidence: float,
    knowledge_level: float,
    pred_accuracy: float,
    control_success: float,
    lp: float,
    sp: float,
    tp: float,
) -> GameWinDigest:
    """Build a compact digest from finished game state (no hidden deps)."""
    all_traces = list(all_traces)
    win_prims = _primitives_in(all_traces)

    tried = set(profiler_stats_keys)
    useful_primitives = sorted(win_prims)
    useless_primitives = sorted(tried - win_prims - {"RESET"})

    useful_op_ids: List[str] = []
    useful_op_kinds: set[str] = set()
    confirmed_op_ids: List[str] = []
    useless_op_kinds: set[str] = set()

    for op_id, op in operators.items():
        conf = float(getattr(op, "confidence", 0.0))
        support = int(getattr(op, "support", 0))
        contradictions = int(getattr(op, "contradictions", 0))
        primitive_name = getattr(op, "primitive_action", None)
        kind = getattr(op, "kind", None)
        # Support enum or str
        kind_str = getattr(kind, "value", str(kind)) if kind is not None else "unknown"

        is_useful = (
            conf >= USEFUL_CONF_MIN
            and support >= USEFUL_SUPPORT_MIN
            and support >= contradictions
        )
        is_useless = (
            support >= USELESS_SUPPORT_MIN
            and (conf <= USELESS_CONF_MAX or contradictions >= 2 * max(support, 1))
        )
        confirmed = primitive_name in win_prims if primitive_name else False

        if is_useful:
            useful_op_ids.append(op_id)
            useful_op_kinds.add(kind_str)
            if confirmed:
                confirmed_op_ids.append(op_id)
        elif is_useless:
            useless_op_kinds.add(kind_str)

    useful_ritual_ids = [
        r.ritual_id
        for r in rituals
        if float(getattr(r, "success_rate", 0.0)) >= USEFUL_RITUAL_SUCCESS_MIN
    ]

    per_level = [len(trace) for trace in all_traces]

    return GameWinDigest(
        game_id=game_id,
        won=won,
        levels_completed=int(levels_completed),
        max_level_reached=int(max_level_reached),
        total_actions=int(total_actions),
        solution_length=int(sum(per_level)),
        per_level_action_counts=per_level[:20],
        ontology=str(top_ontology),
        ontology_confidence=round(float(top_ontology_confidence), 3),
        useful_primitives=useful_primitives[:10],
        useless_primitives=useless_primitives[:10],
        useful_operator_ids=useful_op_ids[:12],
        useful_operator_kinds=sorted(useful_op_kinds)[:8],
        useless_operator_kinds=sorted(useless_op_kinds)[:8],
        confirmed_operator_ids=confirmed_op_ids[:12],
        useful_ritual_ids=useful_ritual_ids[:4],
        knowledge_level=round(float(knowledge_level), 3),
        pred_accuracy=round(float(pred_accuracy), 3),
        control_success=round(float(control_success), 3),
        lp=round(float(lp), 3),
        sp=round(float(sp), 3),
        tp=round(float(tp), 3),
        stop_reason=str(stop_reason),
        elapsed_seconds=round(float(elapsed_seconds), 1),
    )
