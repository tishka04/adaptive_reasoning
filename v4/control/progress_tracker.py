"""Progress tracking and branch-kill logic for V4."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from ..schemas import ObservationV4, TransitionRecord


@dataclass(frozen=True)
class ProgressProfile:
    name: str
    lp_player_displacement_gain: float = 0.05
    lp_predicted_gain: float = 0.07
    lp_click_gain: float = 0.05
    lp_grid_change_gain: float = 0.03
    lp_grid_change_min_cells: int = 4
    lp_decay_no_gain: float = 0.02
    sp_novel_state_gain: float = 0.03
    sp_region_unlock_unit: float = 0.06
    sp_region_unlock_cap: float = 0.16
    sp_class_depletion_gain: float = 0.09
    sp_new_rule_unit: float = 0.04
    sp_new_rule_cap: float = 0.12
    sp_motif_progress_gain: float = 0.05
    sp_structural_change_gain: float = 0.0
    sp_structural_change_min_cells: int = 4
    sp_object_change_gain: float = 0.0
    sp_object_change_min_events: int = 1
    sp_decay_no_gain: float = 0.015
    tp_target_limit: int = 3
    tp_target_narrowing_unit: float = 0.03
    tp_target_narrowing_cap: float = 0.10
    tp_validated_unit: float = 0.08
    tp_validated_cap: float = 0.14
    tp_speculative_unit: float = 0.015
    tp_speculative_cap: float = 0.03
    tp_closure_validated_gain: float = 0.03
    tp_closure_speculative_gain: float = 0.0
    tp_prefix_gain: float = 0.06
    tp_decay_no_gain: float = 0.03
    tp_unsolved_cap_base: float = 0.30
    tp_unsolved_cap_validated_bonus: float = 0.15
    tp_unsolved_cap_prefix_bonus: float = 0.10


def get_progress_profile(name: str) -> ProgressProfile:
    normalized = str(name or "strict").strip().lower()
    if normalized == "strict_plus":
        return ProgressProfile(
            name="strict_plus",
            sp_structural_change_gain=0.02,
            sp_structural_change_min_cells=12,
            sp_object_change_gain=0.02,
            sp_object_change_min_events=2,
        )
    if normalized == "relaxed_sp":
        return ProgressProfile(
            name="relaxed_sp",
            sp_structural_change_gain=0.02,
            sp_structural_change_min_cells=2,
            sp_object_change_gain=0.03,
            sp_object_change_min_events=1,
            sp_decay_no_gain=0.01,
        )
    if normalized == "relaxed_tp":
        return ProgressProfile(
            name="relaxed_tp",
            tp_target_limit=4,
            tp_speculative_unit=0.025,
            tp_speculative_cap=0.05,
            tp_closure_speculative_gain=0.02,
            tp_decay_no_gain=0.02,
            tp_unsolved_cap_base=0.36,
            tp_unsolved_cap_validated_bonus=0.18,
            tp_unsolved_cap_prefix_bonus=0.12,
        )
    return ProgressProfile(name="strict")


@dataclass
class ProgressState:
    lp: float = 0.0
    sp: float = 0.0
    tp: float = 0.0
    branch_id: int = 0
    sterile_repeats: int = 0
    terminal_stall_steps: int = 0


class ProgressTracker:
    """Track local, structural, and terminal progress on multiple timescales."""

    def __init__(self, profile_name: str = "strict_plus") -> None:
        self.state = ProgressState()
        self._profile = get_progress_profile(profile_name)
        self._recent_hashes: deque[int] = deque(maxlen=40)
        self._recent_diffs: deque[str] = deque(maxlen=40)
        self._recent_projects: deque[str] = deque(maxlen=30)
        self._seen_hashes: set[int] = set()
        self._seen_removals: set[int] = set()
        self._smallest_remaining_targets: int = 10_000
        self._last_sp_increase_step: int = 0
        self._last_tp_increase_step: int = 0
        self._steps: int = 0
        self._prev_class_counts: dict[int, int] = {}
        self.num_updates: int = 0
        self.last_update: dict[str, Any] = {}
        self._observed_event_counts: dict[str, Counter[str]] = {
            "lp": Counter(),
            "sp": Counter(),
            "tp": Counter(),
        }
        self._awarded_event_counts: dict[str, Counter[str]] = {
            "lp": Counter(),
            "sp": Counter(),
            "tp": Counter(),
        }
        self._missed_event_counts: dict[str, Counter[str]] = {
            "sp": Counter(),
            "tp": Counter(),
        }

    def set_profile(self, profile_name: str) -> None:
        self._profile = get_progress_profile(profile_name)

    @property
    def profile_name(self) -> str:
        return self._profile.name

    def on_transition(
        self,
        obs: ObservationV4,
        transition: TransitionRecord,
        memory: Any,
    ) -> None:
        self._steps += 1
        self.num_updates += 1
        profile = self._profile
        diff = transition.diff
        prev_lp, prev_sp, prev_tp = self.scores()
        is_reset_action = transition.action.name == "RESET"
        if transition.prev_hash not in self._seen_hashes:
            self._seen_hashes.add(transition.prev_hash)

        def candidate(
            event: str,
            observed: bool,
            awarded: bool,
            amount: float = 0.0,
            detail: Any = None,
            reason: str | None = None,
        ) -> dict[str, Any]:
            return {
                "event": event,
                "observed": bool(observed),
                "awarded": bool(awarded),
                "amount": round(float(amount), 4),
                "detail": detail,
                "reason": reason,
            }

        lp_delta = 0.0
        lp_candidates: list[dict[str, Any]] = []
        player_displacement_observed = (
            not is_reset_action and diff.player_displacement is not None and not diff.game_over
        )
        if player_displacement_observed:
            lp_delta += profile.lp_player_displacement_gain
        lp_candidates.append(
            candidate(
                "player_displacement",
                player_displacement_observed,
                player_displacement_observed,
                profile.lp_player_displacement_gain if player_displacement_observed else 0.0,
                detail=diff.player_displacement,
            )
        )

        predicted_ok_observed = not is_reset_action and bool(transition.metadata.get("predicted_ok"))
        if predicted_ok_observed:
            lp_delta += profile.lp_predicted_gain
        lp_candidates.append(
            candidate(
                "predicted_ok",
                predicted_ok_observed,
                predicted_ok_observed,
                profile.lp_predicted_gain if predicted_ok_observed else 0.0,
            )
        )

        effective_click_observed = not is_reset_action and transition.action.name == "ACTION6" and diff.num_changed > 0
        if effective_click_observed:
            lp_delta += profile.lp_click_gain
        lp_candidates.append(
            candidate(
                "effective_click",
                effective_click_observed,
                effective_click_observed,
                profile.lp_click_gain if effective_click_observed else 0.0,
            )
        )

        meaningful_grid_observed = (
            not is_reset_action and diff.num_changed >= profile.lp_grid_change_min_cells
        )
        if meaningful_grid_observed:
            lp_delta += profile.lp_grid_change_gain
        lp_candidates.append(
            candidate(
                "meaningful_grid_change",
                meaningful_grid_observed,
                meaningful_grid_observed,
                profile.lp_grid_change_gain if meaningful_grid_observed else 0.0,
                detail=diff.num_changed,
            )
        )

        sp_delta = 0.0
        sp_candidates: list[dict[str, Any]] = []
        missed_sp: list[dict[str, Any]] = []

        novel_state_observed = (
            not is_reset_action
            and transition.next_hash not in self._seen_hashes
            and (transition.next_hash != transition.prev_hash or not diff.is_noop)
        )
        if novel_state_observed:
            self._seen_hashes.add(transition.next_hash)
            sp_delta += profile.sp_novel_state_gain
        sp_candidates.append(
            candidate(
                "novel_state",
                novel_state_observed,
                novel_state_observed,
                profile.sp_novel_state_gain if novel_state_observed else 0.0,
            )
        )

        region_unlock_count = len(obs.topology.unlocked_regions) if not is_reset_action else 0
        region_unlock_observed = region_unlock_count > 0
        region_unlock_gain = min(
            profile.sp_region_unlock_cap,
            profile.sp_region_unlock_unit * region_unlock_count,
        ) if region_unlock_observed else 0.0
        if region_unlock_observed:
            sp_delta += region_unlock_gain
        sp_candidates.append(
            candidate(
                "region_unlock",
                region_unlock_observed,
                region_unlock_observed,
                region_unlock_gain,
                detail=region_unlock_count,
            )
        )

        class_counts: dict[int, int] = {}
        for obj in obs.objects:
            class_counts[obj.value] = class_counts.get(obj.value, 0) + 1
        for value, prev_count in self._prev_class_counts.items():
            cur_count = class_counts.get(value, 0)
            depletion_observed = not is_reset_action and cur_count < prev_count and value != 0
            depletion_awarded = depletion_observed and value not in self._seen_removals
            if depletion_awarded:
                self._seen_removals.add(value)
                sp_delta += profile.sp_class_depletion_gain
            detail = {"value": value, "prev": prev_count, "cur": cur_count}
            sp_candidates.append(
                candidate(
                    "class_depletion",
                    depletion_observed,
                    depletion_awarded,
                    profile.sp_class_depletion_gain if depletion_awarded else 0.0,
                    detail=detail,
                    reason=None if depletion_awarded or not depletion_observed else "already_seen_removal",
                )
            )
            if depletion_observed and not depletion_awarded:
                missed_sp.append(
                    {
                        "event": "class_depletion",
                        "reason": "already_seen_removal",
                        "detail": detail,
                    }
                )
        self._prev_class_counts = class_counts

        new_rules_observed = not is_reset_action and int(transition.metadata.get("new_rules", 0)) > 0
        new_rules_gain = min(
            profile.sp_new_rule_cap,
            profile.sp_new_rule_unit * int(transition.metadata.get("new_rules", 0)),
        ) if new_rules_observed else 0.0
        if new_rules_observed:
            sp_delta += new_rules_gain
        sp_candidates.append(
            candidate(
                "new_rules",
                new_rules_observed,
                new_rules_observed,
                new_rules_gain,
                detail=int(transition.metadata.get("new_rules", 0)),
            )
        )

        motif_progress_observed = not is_reset_action and bool(transition.metadata.get("motif_progress"))
        if motif_progress_observed:
            sp_delta += profile.sp_motif_progress_gain
        sp_candidates.append(
            candidate(
                "motif_progress",
                motif_progress_observed,
                motif_progress_observed,
                profile.sp_motif_progress_gain if motif_progress_observed else 0.0,
            )
        )

        structural_change_observed = (
            not is_reset_action
            and not diff.is_noop
            and diff.num_changed >= profile.sp_structural_change_min_cells
        )
        structural_change_awarded = structural_change_observed and profile.sp_structural_change_gain > 0.0
        if structural_change_awarded:
            sp_delta += profile.sp_structural_change_gain
        sp_candidates.append(
            candidate(
                "structural_change",
                structural_change_observed,
                structural_change_awarded,
                profile.sp_structural_change_gain if structural_change_awarded else 0.0,
                detail=diff.num_changed,
                reason=None if structural_change_awarded or not structural_change_observed else "profile_does_not_count_event",
            )
        )
        if structural_change_observed and not structural_change_awarded:
            missed_sp.append(
                {
                    "event": "structural_change",
                    "reason": "profile_does_not_count_event",
                    "detail": {"num_changed": diff.num_changed},
                }
            )

        object_change_count = (
            len(diff.created_object_ids) + len(diff.removed_object_ids) + len(diff.moved_objects)
        )
        object_change_observed = (
            not is_reset_action
            and object_change_count >= profile.sp_object_change_min_events
        )
        object_change_awarded = object_change_observed and profile.sp_object_change_gain > 0.0
        if object_change_awarded:
            sp_delta += profile.sp_object_change_gain
        sp_candidates.append(
            candidate(
                "object_change",
                object_change_observed,
                object_change_awarded,
                profile.sp_object_change_gain if object_change_awarded else 0.0,
                detail=object_change_count,
                reason=None if object_change_awarded or not object_change_observed else "profile_does_not_count_event",
            )
        )
        if object_change_observed and not object_change_awarded:
            missed_sp.append(
                {
                    "event": "object_change",
                    "reason": "profile_does_not_count_event",
                    "detail": {"object_changes": object_change_count},
                }
            )

        if (
            not is_reset_action
            and diff.num_changed > 0
            and not any(item["awarded"] for item in sp_candidates)
        ):
            missed_sp.append(
                {
                    "event": "grid_change_without_sp",
                    "reason": "no_structural_rule_matched",
                    "detail": {"num_changed": diff.num_changed},
                }
            )

        tp_delta = 0.0
        tp_candidates: list[dict[str, Any]] = []
        missed_tp: list[dict[str, Any]] = []

        remaining_targets = sum(1 for obj in obs.objects if obj.value != 0 and obj.area <= 12)
        validated_hits = int(transition.metadata.get("validated_teleology_hits", 0))
        speculative_hits = int(transition.metadata.get("speculative_teleology_hits", 0))
        target_narrowing_observed = (
            not is_reset_action
            and remaining_targets < self._smallest_remaining_targets
            and remaining_targets <= profile.tp_target_limit
        )
        target_narrowing_evidence = bool(
            transition.metadata.get("removed_values")
            or obs.topology.unlocked_regions
            or validated_hits > 0
            or (speculative_hits > 0 and profile.tp_speculative_unit > 0.02)
        )
        target_narrowing_awarded = target_narrowing_observed and target_narrowing_evidence
        target_narrowing_gain = 0.0
        if target_narrowing_awarded:
            improvement = self._smallest_remaining_targets - remaining_targets
            self._smallest_remaining_targets = remaining_targets
            target_narrowing_gain = min(
                profile.tp_target_narrowing_cap,
                profile.tp_target_narrowing_unit * max(improvement, 1),
            )
            tp_delta += target_narrowing_gain
        elif target_narrowing_observed:
            missed_tp.append(
                {
                    "event": "narrowing_targets",
                    "reason": "insufficient_supporting_evidence",
                    "detail": {
                        "remaining_targets": remaining_targets,
                        "validated_hits": validated_hits,
                        "speculative_hits": speculative_hits,
                    },
                }
            )
        tp_candidates.append(
            candidate(
                "narrowing_targets",
                target_narrowing_observed,
                target_narrowing_awarded,
                target_narrowing_gain,
                detail=remaining_targets,
                reason=None if target_narrowing_awarded or not target_narrowing_observed else "insufficient_supporting_evidence",
            )
        )

        level_complete_observed = transition.level_completed and not is_reset_action
        if level_complete_observed:
            tp_delta += 1.0
        tp_candidates.append(
            candidate(
                "level_completed",
                level_complete_observed,
                level_complete_observed,
                1.0 if level_complete_observed else 0.0,
            )
        )

        validated_observed = not is_reset_action and validated_hits > 0
        validated_gain = min(profile.tp_validated_cap, profile.tp_validated_unit * validated_hits) if validated_observed else 0.0
        if validated_observed:
            tp_delta += validated_gain
        tp_candidates.append(
            candidate(
                "validated_teleology",
                validated_observed,
                validated_observed,
                validated_gain,
                detail=validated_hits,
            )
        )

        speculative_observed = not is_reset_action and speculative_hits > 0
        speculative_awarded = speculative_observed and profile.tp_speculative_cap > 0.0
        speculative_gain = min(
            profile.tp_speculative_cap,
            profile.tp_speculative_unit * speculative_hits,
        ) if speculative_awarded else 0.0
        if speculative_awarded:
            tp_delta += speculative_gain
        tp_candidates.append(
            candidate(
                "speculative_teleology",
                speculative_observed,
                speculative_awarded,
                speculative_gain,
                detail=speculative_hits,
            )
        )

        closure_signal_observed = not is_reset_action and bool(transition.metadata.get("closure_signal"))
        closure_validated_awarded = closure_signal_observed and validated_hits > 0 and profile.tp_closure_validated_gain > 0.0
        if closure_validated_awarded:
            tp_delta += profile.tp_closure_validated_gain
        tp_candidates.append(
            candidate(
                "closure_signal_validated",
                closure_signal_observed and validated_hits > 0,
                closure_validated_awarded,
                profile.tp_closure_validated_gain if closure_validated_awarded else 0.0,
            )
        )

        closure_speculative_observed = closure_signal_observed and speculative_hits > 0
        closure_speculative_awarded = closure_speculative_observed and profile.tp_closure_speculative_gain > 0.0
        if closure_speculative_awarded:
            tp_delta += profile.tp_closure_speculative_gain
        tp_candidates.append(
            candidate(
                "closure_signal_speculative",
                closure_speculative_observed,
                closure_speculative_awarded,
                profile.tp_closure_speculative_gain if closure_speculative_awarded else 0.0,
                reason=None if closure_speculative_awarded or not closure_speculative_observed else "profile_does_not_count_event",
            )
        )
        if closure_speculative_observed and not closure_speculative_awarded:
            missed_tp.append(
                {
                    "event": "closure_signal_speculative",
                    "reason": "profile_does_not_count_event",
                    "detail": {"speculative_hits": speculative_hits},
                }
            )

        prefix_replayed_observed = not is_reset_action and bool(transition.metadata.get("prefix_replayed"))
        if prefix_replayed_observed:
            tp_delta += profile.tp_prefix_gain
        tp_candidates.append(
            candidate(
                "prefix_replayed",
                prefix_replayed_observed,
                prefix_replayed_observed,
                profile.tp_prefix_gain if prefix_replayed_observed else 0.0,
            )
        )

        self.state.lp = max(
            0.0,
            min(1.0, self.state.lp * 0.88 + lp_delta - (profile.lp_decay_no_gain if lp_delta <= 0.0 else 0.0)),
        )
        self.state.sp = max(
            0.0,
            min(1.0, self.state.sp * 0.92 + sp_delta - (profile.sp_decay_no_gain if sp_delta <= 0.0 else 0.0)),
        )
        self.state.tp = max(
            0.0,
            min(1.0, self.state.tp * 0.90 + tp_delta - (profile.tp_decay_no_gain if tp_delta <= 0.0 else 0.0)),
        )
        if not transition.level_completed:
            cap = profile.tp_unsolved_cap_base
            if validated_hits > 0:
                cap += profile.tp_unsolved_cap_validated_bonus
            if transition.metadata.get("prefix_replayed"):
                cap += profile.tp_unsolved_cap_prefix_bonus
            self.state.tp = min(self.state.tp, cap)

        if self.state.sp > prev_sp + 0.01:
            self._last_sp_increase_step = self._steps
        if self.state.tp > prev_tp + 0.01:
            self._last_tp_increase_step = self._steps
            self.state.terminal_stall_steps = 0
        else:
            self.state.terminal_stall_steps += 1

        self._recent_hashes.append(transition.next_hash)
        diff_sig = (
            f"{diff.num_changed}:"
            f"{len(diff.created_object_ids)}:"
            f"{len(diff.removed_object_ids)}:"
            f"{diff.player_displacement}"
        )
        self._recent_diffs.append(diff_sig)
        if transition.metadata.get("project_id"):
            self._recent_projects.append(str(transition.metadata["project_id"]))

        repeated_hash_pressure = 0.0
        if self._recent_hashes:
            repeated_hash_pressure = max(
                list(self._recent_hashes).count(h) / len(self._recent_hashes)
                for h in set(self._recent_hashes)
            )
        repeated_diff_pressure = 0.0
        if self._recent_diffs:
            repeated_diff_pressure = max(
                list(self._recent_diffs).count(sig) / len(self._recent_diffs)
                for sig in set(self._recent_diffs)
            )
        self.state.sterile_repeats = int(repeated_hash_pressure * 100)

        self._record_event_stats("lp", lp_candidates, [])
        self._record_event_stats("sp", sp_candidates, missed_sp)
        self._record_event_stats("tp", tp_candidates, missed_tp)

        self.last_update = {
            "step": self._steps,
            "profile": profile.name,
            "action": transition.action.name,
            "project_id": transition.metadata.get("project_id"),
            "lp_before": round(prev_lp, 4),
            "sp_before": round(prev_sp, 4),
            "tp_before": round(prev_tp, 4),
            "lp_after": round(self.state.lp, 4),
            "sp_after": round(self.state.sp, 4),
            "tp_after": round(self.state.tp, 4),
            "lp_delta": round(lp_delta, 4),
            "sp_delta": round(sp_delta, 4),
            "tp_delta": round(tp_delta, 4),
            "lp_candidates": lp_candidates,
            "sp_candidates": sp_candidates,
            "tp_candidates": tp_candidates,
            "missed_sp": missed_sp,
            "missed_tp": missed_tp,
            "validated_hits": validated_hits,
            "speculative_hits": speculative_hits,
            "remaining_targets": remaining_targets,
            "branch_id": self.state.branch_id,
            "repeat_hash_pressure": round(repeated_hash_pressure, 4),
            "repeat_diff_pressure": round(repeated_diff_pressure, 4),
            "recent_unique_states": len(set(self._recent_hashes)),
            "terminal_stall_steps": self.state.terminal_stall_steps,
            "kill_branch": self.should_kill_branch(),
        }

    def scores(self) -> tuple[float, float, float]:
        return self.state.lp, self.state.sp, self.state.tp

    def should_kill_branch(self) -> bool:
        if len(self._recent_hashes) < 20:
            return False

        max_hash_repeat = max(
            list(self._recent_hashes).count(h) for h in set(self._recent_hashes)
        )
        max_diff_repeat = max(
            list(self._recent_diffs).count(sig) for sig in set(self._recent_diffs)
        )
        unique_hashes = len(set(self._recent_hashes))

        no_sp_recent = self._steps - self._last_sp_increase_step >= 40
        no_tp_recent = self._steps - self._last_tp_increase_step >= 50
        repeating_states = max_hash_repeat / max(len(self._recent_hashes), 1) > 0.40
        repeating_diffs = max_diff_repeat / max(len(self._recent_diffs), 1) > 0.50

        return (no_sp_recent and no_tp_recent and (repeating_states or repeating_diffs)) or (
            unique_hashes <= 3 and len(self._recent_hashes) >= 30
        )

    def start_new_branch(self) -> None:
        self.state.branch_id += 1
        self.state.sterile_repeats = 0
        self.state.terminal_stall_steps = 0
        self._recent_hashes.clear()
        self._recent_diffs.clear()
        self._recent_projects.clear()
        self._last_sp_increase_step = self._steps
        self._last_tp_increase_step = self._steps

    def summary(self) -> dict[str, Any]:
        lp, sp, tp = self.scores()
        return {
            "profile": self._profile.name,
            "lp": round(lp, 3),
            "sp": round(sp, 3),
            "tp": round(tp, 3),
            "branch_id": self.state.branch_id,
            "sterile_repeats": self.state.sterile_repeats,
            "terminal_stall_steps": self.state.terminal_stall_steps,
            "recent_unique_states": len(set(self._recent_hashes)),
            "num_updates": self.num_updates,
            "observed_event_counts": {
                channel: dict(counter)
                for channel, counter in self._observed_event_counts.items()
            },
            "awarded_event_counts": {
                channel: dict(counter)
                for channel, counter in self._awarded_event_counts.items()
            },
            "missed_event_counts": {
                channel: dict(counter)
                for channel, counter in self._missed_event_counts.items()
            },
        }

    def diagnostics(self) -> dict[str, Any]:
        return dict(self.last_update)

    def _record_event_stats(
        self,
        channel: str,
        candidates: list[dict[str, Any]],
        missed: list[dict[str, Any]],
    ) -> None:
        observed_counter = self._observed_event_counts[channel]
        awarded_counter = self._awarded_event_counts[channel]
        for item in candidates:
            if item.get("observed"):
                observed_counter[str(item["event"])] += 1
            if item.get("awarded"):
                awarded_counter[str(item["event"])] += 1
        missed_counter = self._missed_event_counts.get(channel)
        if missed_counter is not None:
            for item in missed:
                missed_counter[str(item["event"])] += 1
