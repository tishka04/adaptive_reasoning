"""V5 Adaptive Reasoning Agent.

Structure:
    V5 = V3 spine (inherited wholesale)
       + V4 ontology competition
       + V4 dissent controller (skeptical interrupter)
       + V4 ritualizer + dedicated ritual store
       + V5 goal skeleton (tiny symbolic subgoal hint)
       + V5 digest-centric cross-game memory

This class *inherits* from V3's agent and overrides a few hook points:
  - `_record_transition` — after V3's bookkeeping, update ontology and dissent
  - `_decide`            — before V3's experiment/search/exploit logic, let
                            the dissent controller interrupt if warranted
                            and let the goal skeleton force minds
  - `_on_level_complete` — after V3's solution shortener/macro compiler,
                            compile a ritual and add it to the ritual store
  - `end_game`           — build a compact digest, export to V5 memory

Nothing trains online. The VC adapter is a staged addition (see
`VisualCortexAdapter`) and is OFF by default.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

# Import V3 spine (copied into v5/) — package-relative
from .adaptive_reasoning_agent_v3 import (
    AdaptiveReasoningAgentV3 as _V3Base,
    TIME_BUDGET,
)

from .compression.ritualizer import compile_ritual
from .control.anti_attractor import AntiAttractor
from .control.bissociation_lite import BissociationLite
from .control.danger_memory import DangerMemoryV5
from .control.dissent_controller import DissentController
from .control.goal_skeleton import refresh_goal, mind_bias_for_subgoal
from .memory.cross_game_memory import CrossGameMemoryV5
from .memory.ritual_store import RitualStore
from .memory.win_digest import build_digest
from .ontology.ontology_competition import ONTOLOGY_KINDS, OntologyCompetition
from .schemas import GameObservation, PrimitiveAction
from .schemas_ext import GoalSkeleton, OntologyHypothesis

if TYPE_CHECKING:
    from .control.learned_priors import LearnedPriors

logger = logging.getLogger(__name__)


class AdaptiveReasoningAgentV5(_V3Base):
    """V5 agent: V3 spine + V4 ontology/dissent/ritual + digest memory."""

    def __init__(
        self,
        cross_game: Optional[CrossGameMemoryV5] = None,
        time_budget: float = TIME_BUDGET,
        *,
        use_dissent: bool = True,
        use_ontology: bool = True,
        use_rituals: bool = True,
        use_goal_skeleton: bool = True,
        use_bissociation: bool = True,
        use_danger_memory: bool = True,
        use_anti_attractor: bool = True,
        use_learned_priors: bool = False,
        learned_priors: Optional["LearnedPriors"] = None,
    ) -> None:
        # ── V3 init (without its cross-game seeding) ──
        # We call super().__init__ with cross_game=None and do V5 seeding
        # ourselves afterwards, because V3's seed is incompatible with V5 memory.
        super().__init__(cross_game=None, time_budget=time_budget)

        # Ablation switches
        self._use_dissent = bool(use_dissent)
        self._use_ontology = bool(use_ontology)
        self._use_rituals = bool(use_rituals)
        self._use_goal_skeleton = bool(use_goal_skeleton)
        self._use_bissociation = bool(use_bissociation)
        self._use_danger_memory = bool(use_danger_memory)
        self._use_anti_attractor = bool(use_anti_attractor)
        self._use_learned_priors = bool(
            use_learned_priors and learned_priors is not None
        )
        self.learned_priors = learned_priors if self._use_learned_priors else None
        if use_learned_priors and learned_priors is None:
            logger.warning("Learned priors requested but unavailable; using baseline V5")
        if self._use_learned_priors:
            # Keep the model adapter and its heavy dependencies outside normal
            # V5 startup when the feature is disabled.
            from .control.learned_priors import LearnedPriors

            if not isinstance(self.learned_priors, LearnedPriors):
                raise TypeError("learned_priors must be a LearnedPriors instance")
            self.arbiter.set_prior(
                self._arbiter_prior_bonus,
                self.learned_priors.band,
            )

        # V5 cross-game memory (digest-centric)
        self.cross_game_v5: Optional[CrossGameMemoryV5] = cross_game

        # V5 state
        self.ritual_store = RitualStore(max_active=6) if self._use_rituals else None
        if self._use_ontology:
            priors: Dict[str, float] = {}
            if cross_game is not None:
                priors = dict(cross_game.ontology_priors)
            self.ontology = OntologyCompetition(priors=priors)
        else:
            self.ontology = None
        self.dissent = DissentController() if self._use_dissent else None
        self.bissociation = (
            BissociationLite() if self._use_bissociation else None
        )
        self.goal_skeleton: Optional[GoalSkeleton] = None

        # V5 reactive organs (ported from the greedy controller). Observation-
        # dominated: a hard lethal/no-op veto that gates the chosen action, plus
        # a preventive escape that breaks local attractors before a long stall.
        self.danger_memory: Optional[DangerMemoryV5] = (
            DangerMemoryV5() if self._use_danger_memory else None
        )
        self.anti_attractor: Optional[AntiAttractor] = (
            AntiAttractor() if self._use_anti_attractor else None
        )
        self._danger_vetoes: int = 0
        self._noop_bans: int = 0
        self._escape_steps: int = 0
        self._guard_overrides: int = 0

        # Bissociation-probe state
        self._pending_probe: List[PrimitiveAction] = []
        self._bissociation_probes_emitted: int = 0

        # Dissent can emit a primitive the V3 agent must honour next step.
        self._pending_redirect: Optional[PrimitiveAction] = None
        self._redirect_count: int = 0

        # Diagnostics
        self._ontology_history: List[tuple[str, float]] = []
        self._dissent_interrupts: int = 0
        self._v5_mode_counts: Dict[str, int] = {
            "ontology_active": 0,
            "dissent_interrupt": 0,
            "goal_closure": 0,
            "goal_navigate": 0,
            "goal_explore": 0,
        }

        # Seed V5 cross-game memory (rituals, ontology priors already seeded above)
        if cross_game is not None:
            try:
                cross_game.seed_game(
                    self.memory,
                    ontology_competition=self.ontology,
                    ritual_store=self.ritual_store,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("V5 cross-game seed failed: %s", exc)

    # ------------------------------------------------------------------
    # Transition recording — hook V5 layers after V3 bookkeeping
    # ------------------------------------------------------------------
    def _record_transition(
        self, obs: GameObservation, game_state: str
    ) -> None:
        super()._record_transition(obs, game_state)

        # --- V5 organs: feed danger memory + anti-attractor (observation-only).
        # Runs independently of ontology/dissent so it works under any ablation.
        diff0 = obs.frame_diff
        if (
            (self._use_danger_memory or self._use_anti_attractor)
            and diff0 is not None
            and self._level_action_trace
            and self._prev_obs is not None
        ):
            last_action = self._level_action_trace[-1]
            prev_hash = self._prev_obs.grid_hash  # state before last_action
            if self._use_anti_attractor and self.anti_attractor is not None:
                self.anti_attractor.observe(
                    grid_hash=obs.grid_hash,
                    action_name=last_action.name,
                    is_noop=diff0.is_noop,
                )
                if diff0.is_noop:
                    self.anti_attractor.note_no_effect(prev_hash, last_action.name)
            if (
                self._use_danger_memory
                and self.danger_memory is not None
                and diff0.game_over
            ):
                self.danger_memory.record_primitive_death(prev_hash, last_action)

        diff = obs.frame_diff
        if diff is None or self.ontology is None and self.dissent is None:
            return

        # Last action's x for ontology evidence
        last_x = None
        if self._level_action_trace:
            last_x = self._level_action_trace[-1].x

        # Per-kind operator counts (for ontology evidence)
        op_kinds_seen: Dict[str, int] = {}
        for op in self.inducer.operators.values():
            kind_str = op.kind.value if hasattr(op.kind, "value") else str(op.kind)
            op_kinds_seen[kind_str] = op_kinds_seen.get(kind_str, 0) + 1

        if self._use_ontology and self.ontology is not None:
            self.ontology.update(
                obs,
                diff=diff,
                last_action_x=last_x,
                operator_kinds_seen=op_kinds_seen,
            )
            top = self.ontology.top()
            self._ontology_history.append((top.kind, round(top.confidence, 3)))
            if len(self._ontology_history) > 400:
                self._ontology_history = self._ontology_history[-400:]

        # Dissent report (read-only here; interrupts happen in _decide)
        if self._use_dissent and self.dissent is not None:
            lp, sp, tp = self.progress.scores()
            self.dissent.update(
                obs,
                action_counter=self._action_counter,
                lp=lp,
                sp=sp,
                tp=tp,
                top_ontology=self.ontology.top() if self.ontology is not None else None,
                branch_kill_flag=bool(self.progress.should_kill_branch()),
            )

            # Ontology flip: if dissent suggests it AND we have budget, force a
            # non-current ontology for a short window. This clamps .top() only.
            if (
                self.ontology is not None
                and self.dissent.suggest_ontology_flip(self._action_counter)
            ):
                current = self.ontology.top().kind
                alt = self._pick_alt_ontology(current)
                if alt is not None:
                    self.dissent.force_ontology(alt, self._action_counter)
                    self.ontology.clamp_top(alt)
                    self._v5_mode_counts["ontology_flip"] = (
                        self._v5_mode_counts.get("ontology_flip", 0) + 1
                    )
                    logger.info(
                        "V5 ontology flip: %s -> %s (action=%d)",
                        current, alt, self._action_counter,
                    )

            # Release clamp when the forced window expires
            active_forced = self.dissent.forced_ontology_active(self._action_counter)
            if (
                active_forced is None
                and self.ontology is not None
                and self.ontology.clamped_kind is not None
            ):
                self.ontology.clear_clamp()

        # Bissociation observes SP for stagnation tracking
        if self._use_bissociation and self.bissociation is not None:
            _, sp_now, _ = self.progress.scores()
            self.bissociation.observe(
                action_counter=self._action_counter, sp=sp_now,
            )

    # ------------------------------------------------------------------
    # Decision — let dissent / goal_skeleton intervene before V3 logic
    # ------------------------------------------------------------------
    def _decide(self, obs: GameObservation) -> PrimitiveAction:
        # Anti-attractor channel 2 (preventive escape): fire BEFORE the regular
        # logic when the agent is looping / stalling, so it can break out before
        # burning the whole budget in a local attractor.
        if (
            self._use_anti_attractor
            and self.anti_attractor is not None
            and self.anti_attractor.should_escape(self._action_counter)
        ):
            esc = self.anti_attractor.pick_escape_action(
                available=obs.available_actions,
                grid_hash=obs.grid_hash,
                is_lethal=self._is_lethal_name,
            )
            if esc is not None:
                self.anti_attractor.note_escape(self._action_counter)
                self._escape_steps += 1
                self._v5_mode_counts["anti_attractor_escape"] = (
                    self._v5_mode_counts.get("anti_attractor_escape", 0) + 1
                )
                return PrimitiveAction(name=esc)

        if self._use_learned_priors and self.learned_priors is not None:
            try:
                self._prepare_learned_priors(obs)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Learned prior preparation failed for this step: %s",
                    exc,
                )

        # Run the normal V5/V3 decision, then guard the result (danger veto +
        # no-op ban) so a lethal/dead action can never be emitted unchecked.
        primitive = self._decide_inner(obs)
        return self._guard_action(obs, primitive)

    def _prepare_learned_priors(self, obs: GameObservation) -> None:
        from abstraction_dataset_io import make_history_features

        action_history = [action.name for action in self._level_action_trace]
        repeat_count = 0
        if action_history:
            last_action = action_history[-1]
            for action_name in reversed(action_history):
                if action_name != last_action:
                    break
                repeat_count += 1
        steps_since_change = (
            self.anti_attractor.steps_since_change
            if self.anti_attractor is not None
            else 0
        )
        history_features = make_history_features(
            action_history,
            repeat_count,
            steps_since_change,
        )
        self.learned_priors.begin_step(
            obs.raw_grid,
            [
                action
                for action in obs.available_actions
                if str(action).startswith("ACTION")
            ],
            history_features,
        )

    def _arbiter_prior_bonus(self, proposal: Any) -> float:
        if self.learned_priors is None or not proposal.candidate_plan:
            return 0.0
        call = proposal.candidate_plan[0]
        operator = self.inducer.operators.get(call.operator_id)
        action_name = operator.primitive_action if operator is not None else None
        if not action_name:
            action_name = call.args.get("action")
        if hasattr(action_name, "name"):
            action_name = action_name.name
        if not action_name:
            return 0.0
        return self.learned_priors.bonus(str(action_name))

    def _decide_inner(self, obs: GameObservation) -> PrimitiveAction:
        # 1. Honour a pending redirect from the previous step's dissent.
        if self._pending_redirect is not None:
            primitive = self._pending_redirect
            self._pending_redirect = None
            self._redirect_count += 1
            self._v5_mode_counts["dissent_interrupt"] += 1
            return primitive

        # 2. Refresh goal skeleton
        if self._use_goal_skeleton:
            _, _, tp = self.progress.scores()
            top = self.ontology.top() if self.ontology is not None else None
            self.goal_skeleton = refresh_goal(
                obs, top_ontology=top, tp_estimate=tp,
                prev_goal=self.goal_skeleton,
            )
            subgoal = self.goal_skeleton.active_subgoal
            self._v5_mode_counts[f"goal_{subgoal}"] = (
                self._v5_mode_counts.get(f"goal_{subgoal}", 0) + 1
            )
        else:
            subgoal = "explore"

        # 3. Dissent interrupt
        if (
            self._use_dissent
            and self.dissent is not None
            and self.dissent.should_interrupt(self._action_counter)
        ):
            redirect = self.dissent.interrupt_and_redirect(
                obs,
                profiler=self.profiler,
                action_counter=self._action_counter,
                branch_kill_flag=bool(self.progress.should_kill_branch()),
                ontology_downweight_fn=(
                    self.ontology.downweight if self.ontology is not None else None
                ),
                top_ontology_id=(
                    self.ontology.top().ontology_id
                    if self.ontology is not None else None
                ),
            )
            self._dissent_interrupts += 1
            self._v5_mode_counts["dissent_interrupt"] += 1
            if redirect.primitive is not None:
                return redirect.primitive

        # 3.5 Bissociation probe — cross-ontology hybrid action
        if self._use_bissociation and self.bissociation is not None:
            top_kind = (
                self.ontology.top().kind if self.ontology is not None else "unknown"
            )
            # Monoculture reuses dissent report; safe when dissent is on
            mono = False
            if self.dissent is not None:
                mono = bool(self.dissent.last_report.ontology_warnings)

            if self.bissociation.should_probe(
                action_counter=self._action_counter,
                top_ontology_kind=top_kind,
                ontology_monoculture=mono,
            ):
                digests = (
                    self.cross_game_v5.game_digests
                    if self.cross_game_v5 is not None else {}
                )
                probe = self.bissociation.build_probe(
                    action_counter=self._action_counter,
                    top_ontology_kind=top_kind,
                    cross_game_digests=digests,
                    available_actions=obs.available_actions,
                    objects=obs.objects,
                )
                if probe:
                    self._pending_probe = list(probe[1:])  # queue the rest
                    self._bissociation_probes_emitted += 1
                    self._v5_mode_counts["bissociation_probe"] = (
                        self._v5_mode_counts.get("bissociation_probe", 0) + 1
                    )
                    logger.info(
                        "V5 bissociation probe: len=%d first=%s",
                        len(probe), probe[0].name,
                    )
                    return probe[0]

        # 3.6 Continue a queued probe before any other logic
        if self._pending_probe:
            return self._pending_probe.pop(0)

        # 4. Goal-skeleton forced minds (TP closure / click / transform)
        if self._use_goal_skeleton and self.goal_skeleton is not None:
            if subgoal == "closure" and "closure" in self.minds:
                return self._do_forced_mind(obs, "closure")
            if subgoal == "trigger_transform" and "transform" in self.minds:
                # Only bias; still allow normal decide to take over if mind has no plan
                try:
                    return self._do_forced_mind(obs, "transform")
                except Exception:
                    pass

        # 5. Fallback to V3 decision logic
        return super()._decide(obs)

    # ------------------------------------------------------------------
    # Level completion — compile a ritual in addition to V3 actions
    # ------------------------------------------------------------------
    def _on_level_complete(self, levels_completed: int) -> None:
        trace_copy = list(self._level_action_trace)
        super()._on_level_complete(levels_completed)

        if not self._use_rituals or self.ritual_store is None or not trace_copy:
            return
        top_kind = (
            self.ontology.top().kind if self.ontology is not None else "navigator"
        )
        # Use the solution shortener's output if available; else raw trace
        shortened = trace_copy
        try:
            solved = self.memory.solved_trajectories.get(levels_completed - 1)
            if solved is not None and solved.primitive_actions:
                shortened = solved.primitive_actions
        except Exception:
            pass
        ritual = compile_ritual(
            ritual_id=f"r_{top_kind}_{len(self.ritual_store)}",
            ontology_kind=top_kind,
            successful_prefix=list(shortened),
            current_obs=self._prev_obs if self._prev_obs is not None else _fallback_obs(),
            levels_completed=levels_completed,
        )
        self.ritual_store.add(ritual)
        logger.info(
            "V5 ritual compiled: id=%s kind=%s len=%d",
            ritual.ritual_id, top_kind, len(ritual.prefix),
        )

    # ------------------------------------------------------------------
    # End of game — build digest, export to V5 cross-game
    # ------------------------------------------------------------------
    def end_game(self, won: bool, game_id: str = "unknown") -> Dict[str, Any]:
        # Build V3 summary first (its export_game uses V3 interface — skip it).
        # We inline V3's summary build to avoid calling its cross-game export.
        total = max(self._action_counter, 1)
        prog_summary = self.progress.summary()
        lp, sp, tp = self.progress.scores()

        leverage = {
            "operator_driven_pct": round(100 * self._operator_driven_actions / total, 1),
            "plan_driven_pct": round(100 * self._plan_driven_actions / total, 1),
            "fallback_pct": round(100 * self._fallback_actions / total, 1),
            "mode_counts": dict(self._mode_counts),
            "v5_mode_counts": dict(self._v5_mode_counts),
            "dissent_interrupts": self._dissent_interrupts,
            "profiled_transitions": self.profiler.total_transitions,
            "progress_updates": self.progress.num_updates,
            "induction_ticks": self._induction_ticks,
            "pred_accuracy": round(self.inducer.operator_predictive_accuracy(), 3),
            "control_success": round(self.inducer.operator_control_success(), 3),
            "validated_ops": self.inducer.num_validated(),
            "macros": len(self.memory.macros),
            "progress": prog_summary,
            "compositional": self._has_compositional_evidence(),
            "anti_attractor_escapes": self._escape_steps,
            "danger_vetoes": self._danger_vetoes,
            "noop_bans": self._noop_bans,
            "guard_overrides": self._guard_overrides,
            "prior_reorders": self.arbiter.prior_reorders,
            "prior_promotions": self.arbiter.prior_promotions,
            "danger_memory_size": (
                len(self.danger_memory) if self.danger_memory is not None else 0
            ),
        }

        top_ontology = (
            self.ontology.top() if self.ontology is not None
            else OntologyHypothesis("unknown", "unknown", confidence=0.0)
        )

        # Collect per-level compressed traces for the digest
        all_traces: List[List[PrimitiveAction]] = []
        for traj in self.memory.solved_trajectories.values():
            if traj.primitive_actions:
                all_traces.append(list(traj.primitive_actions))

        rituals = list(self.ritual_store.all()) if self.ritual_store is not None else []

        digest = build_digest(
            game_id=game_id,
            won=won,
            stop_reason="win" if won else "timeout",
            elapsed_seconds=0.0,   # caller (runner) fills this in if needed
            total_actions=self._action_counter,
            levels_completed=self.memory.total_levels_completed,
            max_level_reached=self.memory.max_level_reached,
            operators=self.inducer.operators,
            profiler_stats_keys=self.profiler.stats.keys() if hasattr(self.profiler, "stats") else [],
            all_traces=all_traces,
            rituals=rituals,
            top_ontology=top_ontology.kind,
            top_ontology_confidence=top_ontology.confidence,
            knowledge_level=self.memory.knowledge_level(),
            pred_accuracy=self.inducer.operator_predictive_accuracy(),
            control_success=self.inducer.operator_control_success(),
            lp=lp,
            sp=sp,
            tp=tp,
        )

        # Export to cross-game
        if self.cross_game_v5 is not None:
            self.cross_game_v5.export_digest(digest)
            self.cross_game_v5.export_rituals(rituals)
            self.cross_game_v5.tick_trust(
                won=won,
                pred_accuracy=leverage["pred_accuracy"],
                control_success=leverage["control_success"],
                sp=sp,
                tp=tp,
            )
            # Rebuild derived priors so the next game sees updated weights
            try:
                self.cross_game_v5._rebuild_priors()
            except Exception:
                pass

        summary = {
            "total_actions": self._action_counter,
            "levels_completed": self.memory.total_levels_completed,
            "operators": len(self.inducer.operators),
            "rules": len(self.rule_engine.rules),
            "macros": len(self.memory.macros),
            "rituals": len(rituals),
            "states_visited": len(self._visited_hashes),
            "knowledge_level": self.memory.knowledge_level(),
            "pred_accuracy": leverage["pred_accuracy"],
            "control_success": leverage["control_success"],
            "won": won,
            "mind_selections": dict(self.arbiter._mind_selections),
            "leverage": leverage,
            "progress": prog_summary,
            "ontology": [(k, round(c, 3)) for k, c in (
                self.ontology.summary() if self.ontology is not None else []
            )],
            "dissent_interrupts": self._dissent_interrupts,
            "anti_attractor_escapes": self._escape_steps,
            "danger_vetoes": self._danger_vetoes,
            "noop_bans": self._noop_bans,
            "danger_memory_size": (
                len(self.danger_memory) if self.danger_memory is not None else 0
            ),
            "prior_reorders": self.arbiter.prior_reorders,
            "prior_promotions": self.arbiter.prior_promotions,
            "learned_prior_diagnostics": (
                self.learned_priors.diagnostics()
                if self.learned_priors is not None
                else {}
            ),
            "bissociation_probes": self._bissociation_probes_emitted,
            "ontology_flips_used": (
                self.dissent.ontology_flips_used if self.dissent is not None else 0
            ),
            "digest": digest.to_dict(),
        }
        logger.info("V5 end_game: %s", {k: v for k, v in summary.items() if k != "digest"})
        return summary

    # ------------------------------------------------------------------
    # Reactive guard — danger veto + no-op ban (channel 1)
    # ------------------------------------------------------------------
    def _is_lethal_name(self, grid_hash: int, name: str) -> bool:
        if not self._use_danger_memory or self.danger_memory is None:
            return False
        return self.danger_memory.is_lethal(grid_hash, name)

    def _guard_action(
        self, obs: GameObservation, primitive: PrimitiveAction
    ) -> PrimitiveAction:
        """Veto a chosen action if it is known-lethal or a banned no-op.

        Observation dominates: this overrides whatever the structured logic
        picked. Replacement prefers the anti-attractor escape pool, then any
        available non-lethal / non-banned action.
        """
        gh = obs.grid_hash
        lethal = (
            self._use_danger_memory
            and self.danger_memory is not None
            and self.danger_memory.is_primitive_lethal(gh, primitive)
        )
        banned = (
            self._use_anti_attractor
            and self.anti_attractor is not None
            and self.anti_attractor.is_banned_noop(gh, primitive.name)
        )
        if not lethal and not banned:
            return primitive

        alt: Optional[str] = None
        if self.anti_attractor is not None:
            alt = self.anti_attractor.pick_escape_action(
                available=obs.available_actions,
                grid_hash=gh,
                is_lethal=self._is_lethal_name,
            )
        if alt is None:
            for a in obs.available_actions:
                if a == "RESET" or self._is_lethal_name(gh, a):
                    continue
                if (
                    self._use_anti_attractor
                    and self.anti_attractor is not None
                    and self.anti_attractor.is_banned_noop(gh, a)
                ):
                    continue
                alt = a
                break
        if alt is None or alt == primitive.name:
            return primitive  # nothing strictly better; let it through

        self._guard_overrides += 1
        if lethal:
            self._danger_vetoes += 1
            self._v5_mode_counts["danger_veto"] = (
                self._v5_mode_counts.get("danger_veto", 0) + 1
            )
        if banned:
            self._noop_bans += 1
            self._v5_mode_counts["noop_ban"] = (
                self._v5_mode_counts.get("noop_ban", 0) + 1
            )
        return PrimitiveAction(name=alt)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pick_alt_ontology(self, current: str) -> Optional[str]:
        """Pick a plausible alternative ontology kind to force."""
        if self.ontology is None:
            return None
        ranked = self.ontology.ranked()
        # Prefer the second-best kind that differs from current
        for hyp in ranked:
            if hyp.kind != current:
                return hyp.kind
        # Fall back to any non-current seeded kind
        for kind in ONTOLOGY_KINDS:
            if kind != current:
                return kind
        return None


def _fallback_obs() -> GameObservation:
    """Empty observation used only when a ritual is compiled without a valid prev_obs."""
    return GameObservation(
        raw_grid=np.zeros((1, 1), dtype=np.int32),
        grid_hash=0,
        game_state="NOT_FINISHED",
        levels_completed=0,
        available_actions=[],
    )
