"""V5 Cross-Game Memory — digest-centric.

Primary unit is `GameWinDigest`. Operator / ontology priors are
*derived on load* rather than stored separately, keeping pickled size
small (<100 KB for hundreds of games).

Retained from V3/V4:
  - trust scalar, increased by wins, decayed by poor runs
  - ritual_signatures (compact prefix+terminal templates)

Dropped:
  - raw operator_priors / rule_priors / macro_schemas
    (they are now reconstructed on load from recent digests)
"""

from __future__ import annotations

import logging
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .win_digest import GameWinDigest

logger = logging.getLogger(__name__)


INITIAL_TRUST = 0.15
MAX_TRUST = 0.50
MAX_FILE_SIZE_MB = 5


@dataclass
class CrossGameMemoryV5:
    """Persistent cross-game memory for V5.

    Storage schema:
      game_digests       : dict[game_id -> dict]   # primary
      ritual_signatures  : dict[ritual_id -> dict]
      trust              : float
      games_played       : int
      games_won          : int
      version            : str = "v5"

    Derived at load (not stored):
      operator_kind_priors  : {kind -> prior_weight}
      ontology_priors       : {ontology_kind -> prior_weight}
      primitive_penalties   : {primitive -> penalty}  # for consistently useless primitives
    """

    game_digests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ritual_signatures: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    trust: float = INITIAL_TRUST
    games_played: int = 0
    games_won: int = 0
    version: str = "v5"

    # Derived (populated by _rebuild_priors)
    _operator_kind_priors: Dict[str, float] = field(default_factory=dict, repr=False)
    _ontology_priors: Dict[str, float] = field(default_factory=dict, repr=False)
    _primitive_penalties: Dict[str, float] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Derivation
    # ------------------------------------------------------------------
    def _rebuild_priors(self) -> None:
        """Recompute derived priors from game_digests.

        Weighting policy (partial-credit):
          won            -> 2.0 + 0.5 * (tp + sp)
          tp >= 0.15     -> min(0.8, 0.3 + tp + 0.5 * sp)   # partial
          sp >= 0.30     -> min(0.8, 0.2 + 0.7 * sp)         # partial
          lp >= 0.50     -> 0.2                              # weak signal only
          else           -> 0.0 (ignored for priors)
        """
        op_kind_counter: Counter = Counter()
        op_kind_confirmed: Counter = Counter()
        ontology_counter: Counter = Counter()
        useless_primitive_counter: Counter = Counter()
        partial_credit_games = 0

        for digest in self.game_digests.values():
            won = bool(digest.get("won", False))
            tp = float(digest.get("tp", 0.0) or 0.0)
            sp = float(digest.get("sp", 0.0) or 0.0)
            lp = float(digest.get("lp", 0.0) or 0.0)

            if won:
                weight = 2.0 + 0.5 * (tp + sp)
                ontology_weight = 2.0
            elif tp >= 0.15:
                weight = min(0.8, 0.3 + tp + 0.5 * sp)
                ontology_weight = 0.8
                partial_credit_games += 1
            elif sp >= 0.30:
                weight = min(0.8, 0.2 + 0.7 * sp)
                ontology_weight = 0.5
                partial_credit_games += 1
            elif lp >= 0.50:
                weight = 0.2
                ontology_weight = 0.2
            else:
                weight = 0.0
                ontology_weight = 0.0

            ontology_counter[str(digest.get("ontology", "unknown"))] += ontology_weight

            if weight > 0.0:
                for kind in digest.get("useful_operator_kinds", []) or []:
                    op_kind_counter[str(kind)] += weight
                for op_id in digest.get("confirmed_operator_ids", []) or []:
                    op_kind_confirmed[str(op_id)] += weight

            for prim in digest.get("useless_primitives", []) or []:
                useless_primitive_counter[str(prim)] += 1

        self._partial_credit_games = partial_credit_games

        # Normalize op-kind priors to [0, 1]
        max_op = max(op_kind_counter.values()) if op_kind_counter else 1.0
        self._operator_kind_priors = {
            kind: round(count / max_op, 3)
            for kind, count in op_kind_counter.items()
        }

        # Ontology priors: normalize, clip to [0, 1]
        max_onto = max(ontology_counter.values()) if ontology_counter else 1.0
        self._ontology_priors = {
            kind: round(count / max_onto, 3)
            for kind, count in ontology_counter.items()
            if kind != "unknown"
        }

        # Primitive penalties: a primitive that was useless in 2+ games gets a penalty
        self._primitive_penalties = {
            prim: round(min(0.3, 0.1 * count), 3)
            for prim, count in useless_primitive_counter.items()
            if count >= 2
        }

    @property
    def partial_credit_games(self) -> int:
        return int(getattr(self, "_partial_credit_games", 0))

    @property
    def operator_kind_priors(self) -> Dict[str, float]:
        return self._operator_kind_priors

    @property
    def ontology_priors(self) -> Dict[str, float]:
        return self._ontology_priors

    @property
    def primitive_penalties(self) -> Dict[str, float]:
        return self._primitive_penalties

    # ------------------------------------------------------------------
    # Seeding (import at game start)
    # ------------------------------------------------------------------
    def seed_game(
        self,
        memory: Any,
        *,
        ontology_competition: Any = None,
        ritual_store: Any = None,
    ) -> Dict[str, int]:
        """Seed a new game with trust-gated priors.

        Returns a dict of counts for reporting.
        """
        if self.trust < 0.05:
            logger.info("Cross-game trust too low; skipping seed")
            return {}

        self._rebuild_priors()
        stats = {"rituals": 0, "ontology_priors": 0}

        # Seed ontology priors — injected BEFORE competition is created if
        # caller uses the priors keyword; we just surface them here.
        # (The agent is responsible for applying them via
        # `ontology_competition = OntologyCompetition(priors=self.ontology_priors)`.)
        stats["ontology_priors"] = len(self._ontology_priors)

        # Seed ritual signatures (picks up to 2 with highest success_rate)
        if ritual_store is not None and self.ritual_signatures:
            ranked = sorted(
                self.ritual_signatures.items(),
                key=lambda kv: kv[1].get("success_rate", 0.0),
                reverse=True,
            )
            from ..schemas_ext import Ritual
            from ..schemas import PrimitiveAction
            for ritual_id, data in ranked[:2]:
                prefix_data = data.get("prefix", []) or []
                prefix = [
                    PrimitiveAction(
                        str(p.get("name", "ACTION1")),
                        x=p.get("x"),
                        y=p.get("y"),
                    )
                    for p in prefix_data
                ]
                seeded_success = min(
                    0.35,
                    float(data.get("success_rate", 0.4)) * self.trust,
                )
                ritual = Ritual(
                    ritual_id=ritual_id,
                    ontology_kind=str(data.get("ontology_kind", "navigator")),
                    prefix=prefix,
                    terminal_signature=dict(data.get("terminal_signature", {})),
                    success_rate=seeded_success,
                )
                ritual_store.add(ritual)
                stats["rituals"] += 1

        logger.info(
            f"Cross-game seed: {stats['ontology_priors']} ontology priors, "
            f"{stats['rituals']} rituals, trust={self.trust:.2f}"
        )
        return stats

    # ------------------------------------------------------------------
    # Exporting (import at game end)
    # ------------------------------------------------------------------
    def export_digest(self, digest: GameWinDigest) -> None:
        """Store a digest if it strictly improves the previous one for that game.

        Improvement criteria (in priority order):
          1. first WIN beats any non-WIN
          2. more levels completed
          3. same levels/won + strictly shorter solution
          4. richer partial credit (tp then sp) when neither won
        """
        existing = self.game_digests.get(digest.game_id)
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
            elif (
                not digest.won
                and not prev.won
                and digest.levels_completed == prev.levels_completed
                and (digest.tp > prev.tp + 0.02 or digest.sp > prev.sp + 0.05)
            ):
                better = True
            if not better:
                return
        self.game_digests[digest.game_id] = digest.to_dict()

    def export_rituals(self, rituals: List[Any]) -> None:
        """Store top rituals by success_rate."""
        for ritual in rituals:
            if float(getattr(ritual, "success_rate", 0.0)) < 0.45:
                continue
            self.ritual_signatures[ritual.ritual_id] = {
                "ontology_kind": ritual.ontology_kind,
                "prefix": [
                    {"name": a.name, "x": a.x, "y": a.y} for a in ritual.prefix
                ],
                "terminal_signature": dict(ritual.terminal_signature),
                "success_rate": float(ritual.success_rate),
            }
        # Cap stored rituals
        if len(self.ritual_signatures) > 20:
            ranked = sorted(
                self.ritual_signatures.items(),
                key=lambda kv: kv[1].get("success_rate", 0.0),
                reverse=True,
            )
            self.ritual_signatures = dict(ranked[:20])

    def tick_trust(
        self,
        *,
        won: bool,
        pred_accuracy: float,
        control_success: float,
        sp: float = 0.0,
        tp: float = 0.0,
    ) -> None:
        """Update trust after a game completes.

        Partial credit: games with sp>=0.30 or tp>=0.15 get a small
        positive bump even without a win. Wins still dominate.
        """
        self.games_played += 1
        if won:
            self.games_won += 1
            self.trust = min(MAX_TRUST, self.trust + 0.05)
        elif tp >= 0.15 or sp >= 0.30:
            self.trust = min(MAX_TRUST, self.trust + 0.02)
        elif pred_accuracy >= 0.55 and control_success >= 0.45:
            self.trust = min(MAX_TRUST, self.trust + 0.01)
        else:
            self.trust = max(INITIAL_TRUST, self.trust * 0.97)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        data = {
            "game_digests": self.game_digests,
            "ritual_signatures": self.ritual_signatures,
            "trust": self.trust,
            "games_played": self.games_played,
            "games_won": self.games_won,
            "version": self.version,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            logger.warning(f"V5 cross-game memory is {size_mb:.1f}MB")
        else:
            logger.info(f"Saved V5 cross-game memory: {size_mb:.2f}MB")

    @classmethod
    def load(cls, path: str) -> "CrossGameMemoryV5":
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            logger.info(f"No valid V5 memory at {path}; starting fresh")
            return cls()
        if data.get("version") != "v5":
            logger.info("V5 memory version mismatch; starting fresh")
            return cls()
        m = cls()
        m.game_digests = data.get("game_digests", {})
        m.ritual_signatures = data.get("ritual_signatures", {})
        m.trust = min(MAX_TRUST, data.get("trust", INITIAL_TRUST))
        m.games_played = data.get("games_played", 0)
        m.games_won = data.get("games_won", 0)
        m._rebuild_priors()
        return m


# Backward-compat alias so external code can keep the old name if needed.
CrossGameMemoryV3 = CrossGameMemoryV5  # noqa: E305
