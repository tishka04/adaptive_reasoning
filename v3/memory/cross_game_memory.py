"""V3 Cross-Game Memory — trust-gated transfer of compact abstractions.

Only transfers compact abstractions:
  - Operator templates (not raw action stats)
  - Rule priors
  - Macro schemas
  - Failure motifs
  - Specialist mind reliability priors

All gated by a trust system that starts low and requires in-game validation.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

INITIAL_TRUST = 0.30
COMPETITION_TRUST = 0.15
MAX_TRUST = 0.70
TRUST_DECAY_ON_DISAGREE = 0.50
TRUST_GROWTH_ON_AGREE = 0.10
MAX_FILE_SIZE_MB = 10


@dataclass
class CrossGameMemoryV3:
    """Persistent cross-game memory for V3 architecture."""

    # Operator templates: kind → list of lightweight dicts
    operator_priors: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Rule priors: rule_id → dict
    rule_priors: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Macro schemas: macro_id → dict
    macro_schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Failure patterns: motif_hash → {type, count, trace}
    failure_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Mind reliability: mind_name → recent_accuracy
    mind_reliability: Dict[str, float] = field(default_factory=dict)

    # Episodic graph fragments (compact)
    graph_fragments: Dict[str, Any] = field(default_factory=dict)

    # Trust
    trust: float = INITIAL_TRUST

    # Diagnostics only
    games_played: int = 0
    games_won: int = 0
    mode: str = "development"

    def set_mode(self, mode: str) -> None:
        """Set competition or development mode."""
        self.mode = mode
        if mode == "competition":
            self.trust = COMPETITION_TRUST
        else:
            self.trust = INITIAL_TRUST

    # ─── Import (seeding a new game) ────────────────────────────

    def seed_game(self, memory: Any) -> None:
        """Seed a new game's memory with trust-gated priors.

        Args:
            memory: GameMemoryV3 instance to seed.
        """
        if self.trust < 0.05:
            logger.info("Cross-game trust too low; skipping seed")
            return

        # Seed operator priors
        seeded_ops = 0
        for kind, templates in self.operator_priors.items():
            for tmpl in templates[:3]:  # max 3 per kind
                op_id = tmpl.get("operator_id", "")
                if op_id and op_id not in memory.inducer.operators:
                    from ..schemas import Operator, OperatorKind
                    try:
                        op = Operator(
                            operator_id=op_id,
                            kind=OperatorKind(kind),
                            parameters=tmpl.get("parameters", {}),
                            primitive_action=tmpl.get("primitive_action"),
                            confidence=tmpl.get("confidence", 0.3) * self.trust,
                            support=max(1, int(tmpl.get("support", 3) * self.trust)),
                            contradictions=0,
                        )
                        memory.inducer.operators[op_id] = op
                        seeded_ops += 1
                    except (ValueError, KeyError):
                        pass

        # Seed rule priors
        seeded_rules = 0
        for rid, rdata in self.rule_priors.items():
            if rid not in memory.rules.rules:
                from ..schemas import Rule
                rule = Rule(
                    rule_id=rid,
                    conditions=[],  # simplified — conditions must be re-verified
                    operator_kind=rdata.get("operator_kind", ""),
                    effects=[],     # effects must be re-discovered in this game
                    confidence=rdata.get("confidence", 0.3) * self.trust,
                    support=max(1, int(rdata.get("support", 2) * self.trust)),
                )
                memory.rules.rules[rid] = rule
                seeded_rules += 1

        # Seed mind reliability
        for mind_name, reliability in self.mind_reliability.items():
            # Blend with neutral prior
            blended = 0.5 + (reliability - 0.5) * self.trust
            self.mind_reliability[mind_name] = blended

        logger.info(
            f"Cross-game seed: {seeded_ops} operators, {seeded_rules} rules, "
            f"trust={self.trust:.2f}"
        )

    # ─── Export (after game ends) ───────────────────────────────

    def export_game(
        self,
        memory: Any,
        won: bool,
        mind_stats: Optional[Dict[str, float]] = None,
        graph_export: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Export this game's learnings to cross-game memory.

        Args:
            memory: GameMemoryV3 instance.
            won: Whether the game was won.
            mind_stats: {mind_name: recent_accuracy}.
            graph_export: Compact episodic graph export.
        """
        self.games_played += 1
        if won:
            self.games_won += 1

        # Export high-confidence operators
        for op in memory.inducer.operators.values():
            if op.confidence < 0.5:
                continue
            kind_str = op.kind.value
            tmpl = {
                "operator_id": op.operator_id,
                "parameters": op.parameters,
                "primitive_action": op.primitive_action,
                "confidence": op.confidence,
                "support": op.support,
            }
            if kind_str not in self.operator_priors:
                self.operator_priors[kind_str] = []
            # Avoid duplicates
            existing_ids = {t.get("operator_id") for t in self.operator_priors[kind_str]}
            if op.operator_id not in existing_ids:
                self.operator_priors[kind_str].append(tmpl)
            # Cap per kind
            self.operator_priors[kind_str] = self.operator_priors[kind_str][:5]

        # Export high-confidence rules
        for rule in memory.rules.rules.values():
            if rule.confidence < 0.5:
                continue
            self.rule_priors[rule.rule_id] = {
                "operator_kind": rule.operator_kind,
                "confidence": rule.confidence,
                "support": rule.support,
            }
        # Cap rules
        if len(self.rule_priors) > 30:
            sorted_rules = sorted(
                self.rule_priors.items(),
                key=lambda x: x[1].get("confidence", 0),
                reverse=True,
            )
            self.rule_priors = dict(sorted_rules[:30])

        # Export macros
        for m in memory.macros.values():
            if m.success_rate < 0.5:
                continue
            self.macro_schemas[m.macro_id] = {
                "name": m.name[:80],
                "num_steps": len(m.steps),
                "success_rate": m.success_rate,
            }
        if len(self.macro_schemas) > 20:
            sorted_macros = sorted(
                self.macro_schemas.items(),
                key=lambda x: x[1].get("success_rate", 0),
                reverse=True,
            )
            self.macro_schemas = dict(sorted_macros[:20])

        # Export failure patterns
        for f in memory.failure_patterns:
            self.failure_patterns[f.motif_hash] = {
                "type": f.failure_type,
                "count": f.count,
                "trace": f.operator_trace[:5],
            }
        if len(self.failure_patterns) > 50:
            sorted_fp = sorted(
                self.failure_patterns.items(),
                key=lambda x: x[1].get("count", 0),
                reverse=True,
            )
            self.failure_patterns = dict(sorted_fp[:50])

        # Export mind reliability
        if mind_stats:
            for name, acc in mind_stats.items():
                old = self.mind_reliability.get(name, 0.5)
                self.mind_reliability[name] = 0.7 * old + 0.3 * acc

        # Export graph fragments
        if graph_export:
            self.graph_fragments = graph_export

        logger.info(
            f"Cross-game export: ops={sum(len(v) for v in self.operator_priors.values())}, "
            f"rules={len(self.rule_priors)}, macros={len(self.macro_schemas)}, "
            f"games={self.games_played}/{self.games_won}won"
        )

    # ─── Trust management ───────────────────────────────────────

    def trust_agree(self) -> None:
        self.trust = min(MAX_TRUST, self.trust + TRUST_GROWTH_ON_AGREE)

    def trust_disagree(self) -> None:
        self.trust *= TRUST_DECAY_ON_DISAGREE

    # ─── Persistence ────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save to disk."""
        data = {
            "operator_priors": self.operator_priors,
            "rule_priors": self.rule_priors,
            "macro_schemas": self.macro_schemas,
            "failure_patterns": self.failure_patterns,
            "mind_reliability": self.mind_reliability,
            "graph_fragments": self.graph_fragments,
            "trust": self.trust,
            "games_played": self.games_played,
            "games_won": self.games_won,
            "mode": self.mode,
            "version": "v3",
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        size_mb = Path(path).stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            logger.warning(
                f"Cross-game memory is {size_mb:.1f}MB (> {MAX_FILE_SIZE_MB}MB)!"
            )
        else:
            logger.info(f"Saved cross-game memory: {size_mb:.2f}MB")

    @classmethod
    def load(cls, path: str) -> "CrossGameMemoryV3":
        """Load from disk."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            logger.info(f"No valid cross-game memory at {path}; starting fresh")
            return cls()

        if data.get("version") != "v3":
            logger.info("Cross-game memory version mismatch; starting fresh")
            return cls()

        mem = cls()
        mem.operator_priors = data.get("operator_priors", {})
        mem.rule_priors = data.get("rule_priors", {})
        mem.macro_schemas = data.get("macro_schemas", {})
        mem.failure_patterns = data.get("failure_patterns", {})
        mem.mind_reliability = data.get("mind_reliability", {})
        mem.graph_fragments = data.get("graph_fragments", {})
        mem.trust = data.get("trust", INITIAL_TRUST)
        mem.games_played = data.get("games_played", 0)
        mem.games_won = data.get("games_won", 0)
        mem.mode = data.get("mode", "development")

        logger.info(
            f"Loaded cross-game memory: {mem.games_played} games, "
            f"{mem.games_won} won, trust={mem.trust:.2f}"
        )
        return mem
