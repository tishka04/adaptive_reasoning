"""Operator induction for V4."""

from __future__ import annotations

from typing import Any

from ..schemas import Effect, ObservationV4, Operator, Predicate
from .action_profiler import ActionProfiler


def _sigmoid_confidence(support: int, contradictions: int) -> float:
    total = support + contradictions
    if total <= 0:
        return 0.0
    return (support / total) * min(1.0, support / 8.0)


def _predicate(kind: str, **args: Any) -> Predicate:
    return Predicate(kind=kind, args=args)


def _effect(kind: str, **args: Any) -> Effect:
    return Effect(kind=kind, args=args)


def check_predicate(predicate: Predicate, obs: ObservationV4) -> bool:
    player = obs.best_player
    grid = obs.raw_grid
    if predicate.kind == "player_exists":
        return player is not None and player.confidence > 0.25
    if predicate.kind == "cell_free":
        if player is None:
            return False
        dy = int(predicate.args.get("dy", 0))
        dx = int(predicate.args.get("dx", 0))
        nr = player.position[0] + dy
        nc = player.position[1] + dx
        if nr < 0 or nr >= grid.shape[0] or nc < 0 or nc >= grid.shape[1]:
            return False
        return int(grid[nr, nc]) == 0
    if predicate.kind == "object_exists":
        value = predicate.args.get("value")
        return any(obj.value == value for obj in obs.objects)
    if predicate.kind == "adjacent_value":
        if player is None:
            return False
        value = predicate.args.get("value")
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = player.position[0] + dr
            nc = player.position[1] + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if int(grid[nr, nc]) == value:
                    return True
        return False
    return True


class OperatorInducer:
    """Infer reusable operators from action profiles."""

    def __init__(self) -> None:
        self.operators: dict[str, Operator] = {}
        self._validation: dict[str, dict[str, float]] = {}
        self._induction_count = 0

    def induce(self, profiler: ActionProfiler, memory: Any) -> list[Operator]:
        self._induction_count += 1
        recent = profiler.transitions[-200:]
        new_ops: list[Operator] = []

        for action_name, stats in profiler.stats.items():
            if stats.total_tries < 2:
                continue

            disp = profiler.dominant_displacement(action_name)
            if disp is not None:
                dy, dx = disp
                support = sum(
                    bucket.tries
                    for bucket in stats.context_buckets.values()
                    if (
                        round(bucket.mean_disp[0]) == dy
                        and round(bucket.mean_disp[1]) == dx
                    )
                )
                contradictions = max(0, stats.total_tries - support)
                conf = _sigmoid_confidence(support, contradictions)
                if conf >= 0.35:
                    direction = (
                        "up" if dy < 0 else
                        "down" if dy > 0 else
                        "left" if dx < 0 else "right"
                    )
                    new_ops.append(
                        Operator(
                            operator_id=f"move_{direction}_{action_name}",
                            kind="move",
                            primitive_action=action_name,
                            parameters={"dy": dy, "dx": dx, "direction": direction},
                            preconditions=[_predicate("player_exists"), _predicate("cell_free", dy=dy, dx=dx)],
                            expected_effects=[_effect("player_displacement", delta=(dy, dx))],
                            confidence=conf,
                            support=support,
                            contradictions=contradictions,
                            risk_estimate=stats.death_rate,
                            contexts_supported=["movement"],
                        )
                    )

            noop_support = stats.total_tries - stats.total_changes
            noop_conf = _sigmoid_confidence(noop_support, stats.total_changes)
            if noop_conf >= 0.35 and noop_support >= 2:
                new_ops.append(
                    Operator(
                        operator_id=f"noop_{action_name}",
                        kind="noop",
                        primitive_action=action_name,
                        expected_effects=[_effect("noop")],
                        confidence=noop_conf,
                        support=noop_support,
                        contradictions=stats.total_changes,
                        risk_estimate=stats.death_rate,
                        contexts_supported=["low_change"],
                    )
                )

            lethal_support = stats.deaths
            lethal_conf = _sigmoid_confidence(lethal_support, stats.total_tries - lethal_support)
            if lethal_conf >= 0.35 and lethal_support >= 2:
                new_ops.append(
                    Operator(
                        operator_id=f"lethal_{action_name}",
                        kind="lethal",
                        primitive_action=action_name,
                        expected_effects=[_effect("game_over")],
                        confidence=lethal_conf,
                        support=lethal_support,
                        contradictions=max(0, stats.total_tries - lethal_support),
                        risk_estimate=1.0,
                        contexts_supported=["hazard"],
                    )
                )

            big_changes = 0
            coord_hits = 0
            repeat_until_blocked = 0
            sweep_click = 0
            for index, transition in enumerate(recent):
                if transition.action.name != action_name:
                    continue
                if transition.diff.num_changed >= 8:
                    big_changes += 1
                if transition.action.x is not None and transition.diff.num_changed > 0:
                    coord_hits += 1
                if (
                    transition.diff.num_changed > 0
                    and index + 1 < len(recent)
                    and recent[index + 1].action.name == action_name
                    and recent[index + 1].diff.is_noop
                ):
                    repeat_until_blocked += 1
                if transition.action.x is not None and transition.diff.num_changed > 0:
                    sweep_click += 1

            transform_conf = _sigmoid_confidence(big_changes, max(0, stats.total_tries - big_changes))
            if transform_conf >= 0.35 and big_changes >= 2:
                new_ops.append(
                    Operator(
                        operator_id=f"transform_{action_name}",
                        kind="global_transform",
                        primitive_action=action_name,
                        expected_effects=[_effect("grid_change", min_cells=8)],
                        confidence=transform_conf,
                        support=big_changes,
                        contradictions=max(0, stats.total_tries - big_changes),
                        risk_estimate=stats.death_rate,
                        contexts_supported=["transform"],
                    )
                )

            click_conf = _sigmoid_confidence(coord_hits, max(0, stats.total_tries - coord_hits))
            if click_conf >= 0.35 and coord_hits >= 2:
                target_values: dict[int, int] = {}
                for transition in recent:
                    if transition.action.name != action_name or transition.action.x is None:
                        continue
                    value = int(
                        transition.metadata.get("click_value_before", -1)
                    )
                    if value > 0:
                        target_values[value] = target_values.get(value, 0) + 1
                if target_values:
                    target_value = max(target_values.items(), key=lambda item: item[1])[0]
                    preconditions = [_predicate("object_exists", value=target_value)]
                else:
                    target_value = None
                    preconditions = []
                new_ops.append(
                    Operator(
                        operator_id=f"click_{action_name}_{target_value}",
                        kind="click",
                        primitive_action=action_name,
                        parameters={"target_value": target_value},
                        preconditions=preconditions,
                        expected_effects=[_effect("grid_change", min_cells=1)],
                        confidence=click_conf,
                        support=coord_hits,
                        contradictions=max(0, stats.total_tries - coord_hits),
                        risk_estimate=stats.death_rate,
                        contexts_supported=["click"],
                    )
                )

            repeat_conf = _sigmoid_confidence(repeat_until_blocked, max(0, stats.total_tries - repeat_until_blocked))
            if repeat_conf >= 0.35 and repeat_until_blocked >= 2:
                new_ops.append(
                    Operator(
                        operator_id=f"repeat_until_blocked_{action_name}",
                        kind="repeat_until_blocked",
                        primitive_action=action_name,
                        expected_effects=[_effect("grid_change", min_cells=1)],
                        confidence=repeat_conf,
                        support=repeat_until_blocked,
                        contradictions=max(0, stats.total_tries - repeat_until_blocked),
                        risk_estimate=stats.death_rate,
                        contexts_supported=["repeat"],
                    )
                )

            sweep_conf = _sigmoid_confidence(sweep_click, max(0, stats.total_tries - sweep_click))
            if sweep_conf >= 0.35 and sweep_click >= 3 and stats.coord_uses >= 3:
                new_ops.append(
                    Operator(
                        operator_id=f"sweep_click_{action_name}",
                        kind="sweep_click",
                        primitive_action=action_name,
                        expected_effects=[_effect("grid_change", min_cells=1)],
                        confidence=sweep_conf,
                        support=sweep_click,
                        contradictions=max(0, stats.total_tries - sweep_click),
                        risk_estimate=stats.death_rate,
                        contexts_supported=["click_sweep"],
                    )
                )

        # Escape operators are derived from safest movement operators.
        move_ops = [op for op in new_ops if op.kind == "move" and op.risk_estimate < 0.25]
        for move_op in move_ops[:4]:
            new_ops.append(
                Operator(
                    operator_id=f"escape_{move_op.operator_id}",
                    kind="escape",
                    primitive_action=move_op.primitive_action,
                    parameters=move_op.parameters.copy(),
                    preconditions=move_op.preconditions[:],
                    expected_effects=move_op.expected_effects[:],
                    confidence=min(0.95, move_op.confidence + 0.10),
                    support=move_op.support,
                    contradictions=move_op.contradictions,
                    risk_estimate=max(0.0, move_op.risk_estimate * 0.5),
                    contexts_supported=move_op.contexts_supported + ["escape"],
                )
            )

        for op in new_ops:
            existing = self.operators.get(op.operator_id)
            if existing is None or op.confidence >= existing.confidence:
                self.operators[op.operator_id] = op

        return new_ops

    def get_applicable(self, obs: ObservationV4) -> list[Operator]:
        return [
            op for op in self.operators.values()
            if op.kind != "lethal" and all(check_predicate(p, obs) for p in op.preconditions)
        ]

    def get_movement_ops(self) -> list[Operator]:
        return [op for op in self.operators.values() if op.kind == "move"]

    def get_by_kind(self, kind: str) -> list[Operator]:
        return [op for op in self.operators.values() if op.kind == kind]

    def best_operator_confidence(self) -> float:
        return max((op.confidence for op in self.operators.values()), default=0.0)

    def record_validation(self, operator_id: str, predicted_ok: bool, had_progress: bool) -> None:
        stats = self._validation.setdefault(
            operator_id, {"used": 0.0, "correct": 0.0, "progress": 0.0}
        )
        stats["used"] += 1
        if predicted_ok:
            stats["correct"] += 1
        if had_progress:
            stats["progress"] += 1
        operator = self.operators.get(operator_id)
        if operator is not None:
            operator.survival_score = self._survival_score(operator_id, operator)

    def operator_predictive_accuracy(self) -> float:
        accuracies = [
            stats["correct"] / max(stats["used"], 1.0)
            for stats in self._validation.values()
            if stats["used"] >= 2
        ]
        return sum(accuracies) / len(accuracies) if accuracies else 0.0

    def operator_control_success(self) -> float:
        used = sum(stats["used"] for stats in self._validation.values())
        progress = sum(stats["progress"] for stats in self._validation.values())
        return progress / max(used, 1.0)

    def num_validated(self, min_uses: int = 3, min_accuracy: float = 0.5) -> int:
        count = 0
        for operator_id, stats in self._validation.items():
            accuracy = stats["correct"] / max(stats["used"], 1.0)
            if stats["used"] >= min_uses and accuracy >= min_accuracy:
                if operator_id in self.operators:
                    count += 1
        return count

    def best_validated_confidence(self) -> float:
        best = 0.0
        for operator_id, stats in self._validation.items():
            accuracy = stats["correct"] / max(stats["used"], 1.0)
            if stats["used"] >= 2 and accuracy >= 0.5:
                op = self.operators.get(operator_id)
                if op is not None:
                    best = max(best, op.confidence)
        return best

    def prune(self, max_active: int = 10, memory: Any | None = None) -> list[str]:
        if len(self.operators) <= max_active:
            for operator_id, operator in self.operators.items():
                operator.survival_score = self._survival_score(operator_id, operator, memory=memory)
            return []

        scored = []
        for operator_id, operator in self.operators.items():
            score = self._survival_score(operator_id, operator, memory=memory)
            operator.survival_score = score
            scored.append((score, operator_id, operator))

        keep_ids: list[str] = []
        for kind in ("move", "click", "global_transform", "escape"):
            candidates = [item for item in scored if item[2].kind == kind]
            if candidates:
                keep_ids.append(max(candidates, key=lambda item: item[0])[1])

        for _, operator_id, _ in sorted(scored, key=lambda item: item[0], reverse=True):
            if operator_id not in keep_ids:
                keep_ids.append(operator_id)
            if len(keep_ids) >= max_active:
                break

        removed = [operator_id for operator_id in list(self.operators) if operator_id not in keep_ids]
        for operator_id in removed:
            self.operators.pop(operator_id, None)
        return removed

    def _survival_score(self, operator_id: str, operator: Operator, memory: Any | None = None) -> float:
        stats = self._validation.get(operator_id, {})
        used = float(stats.get("used", 0.0))
        predictive_value = (
            float(stats.get("correct", 0.0)) / used
            if used > 0 else min(0.25, operator.confidence * 0.5)
        )
        control_value = (
            float(stats.get("progress", 0.0)) / used
            if used > 0 else 0.0
        )
        contradiction_rate = operator.contradictions / max(
            operator.support + operator.contradictions, 1
        )
        idle_penalty = 0.15 if used < 2 and operator.support < 3 else 0.0
        score = (
            0.40 * predictive_value
            + 0.30 * control_value
            + 0.20 * operator.confidence
            + 0.10 * min(operator.support / 6.0, 1.0)
            - 0.40 * contradiction_rate
            - idle_penalty
        )
        if operator.kind == "move":
            score += 0.05
        if operator.kind == "click":
            score += 0.03
        if operator.kind == "lethal":
            score -= 0.10
        if memory is not None and getattr(memory.game, "learning", None) is not None:
            score = memory.game.learning.operator_utility.adjust(score, operator, memory)
        return score
