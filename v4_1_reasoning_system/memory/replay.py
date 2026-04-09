"""
Episodic memory — replay buffer for reasoning trajectories.

Stores full reasoning trajectories including:
  - mode choices at each step
  - latent states
  - solver outcomes
  - verifier feedback
  - failures and successful patches

Used for:
  - training the world model and router EBM
  - analyzing failure patterns
  - computing statistics for the candidate generator
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class StepRecord:
    """A single reasoning step within a trajectory."""
    step_idx: int
    mode: str
    budget: str
    strictness: str
    tool_hint: str
    z_t: Optional[torch.Tensor] = None
    action_emb: Optional[torch.Tensor] = None
    z_hat: Optional[torch.Tensor] = None
    z_actual: Optional[torch.Tensor] = None
    solver_result: Optional[Dict[str, Any]] = None
    verifier_result: Optional[Dict[str, Any]] = None
    success: bool = False
    score_before: float = 0.0
    score_after: float = 0.0
    elapsed_seconds: float = 0.0
    candidates_considered: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def score_delta(self) -> float:
        return self.score_after - self.score_before

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "mode": self.mode,
            "budget": self.budget,
            "strictness": self.strictness,
            "tool_hint": self.tool_hint,
            "success": self.success,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "score_delta": self.score_delta,
            "elapsed_seconds": self.elapsed_seconds,
            "solver_result": self.solver_result,
            "verifier_result": self.verifier_result,
        }


@dataclass
class TrajectoryRecord:
    """A complete reasoning trajectory for one problem."""
    trajectory_id: str
    task_summary: str
    domain: str
    steps: List[StepRecord] = field(default_factory=list)
    final_success: bool = False
    final_score: float = 0.0
    total_elapsed: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def modes_used(self) -> List[str]:
        return [s.mode for s in self.steps]

    @property
    def repair_count(self) -> int:
        return sum(1 for s in self.steps if s.mode == "repair")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "task_summary": self.task_summary,
            "domain": self.domain,
            "num_steps": self.num_steps,
            "final_success": self.final_success,
            "final_score": self.final_score,
            "total_elapsed": self.total_elapsed,
            "timestamp": self.timestamp,
            "modes_used": self.modes_used,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
        }


class ReplayBuffer:
    """
    Stores reasoning trajectories for training and analysis.

    Supports:
      - FIFO eviction when capacity is exceeded
      - Sampling by domain, success, or random
      - Persistence to disk (JSON + tensors)
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._buffer: List[TrajectoryRecord] = []
        self._tensor_store: Dict[str, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, trajectory: TrajectoryRecord) -> None:
        """Add a trajectory to the buffer."""
        if len(self._buffer) >= self.capacity:
            evicted = self._buffer.pop(0)
            self._tensor_store.pop(evicted.trajectory_id, None)

        self._buffer.append(trajectory)

        # Store tensors separately for efficient access
        tensors = {}
        for step in trajectory.steps:
            prefix = f"step_{step.step_idx}"
            if step.z_t is not None:
                tensors[f"{prefix}_z_t"] = step.z_t.detach().cpu()
            if step.action_emb is not None:
                tensors[f"{prefix}_action_emb"] = step.action_emb.detach().cpu()
            if step.z_hat is not None:
                tensors[f"{prefix}_z_hat"] = step.z_hat.detach().cpu()
            if step.z_actual is not None:
                tensors[f"{prefix}_z_actual"] = step.z_actual.detach().cpu()
        self._tensor_store[trajectory.trajectory_id] = tensors

    def sample(self, n: int = 1, domain: Optional[str] = None, success_only: bool = False) -> List[TrajectoryRecord]:
        """Sample trajectories from the buffer."""
        pool = self._buffer
        if domain:
            pool = [t for t in pool if t.domain == domain]
        if success_only:
            pool = [t for t in pool if t.final_success]
        if not pool:
            return []
        return random.sample(pool, min(n, len(pool)))

    def get_tensors(self, trajectory_id: str) -> Dict[str, torch.Tensor]:
        """Get stored tensors for a trajectory."""
        return self._tensor_store.get(trajectory_id, {})

    def get_training_pairs(
        self,
        latent_dim: int = 128,
        action_dim: int = 32,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Extract (good, bad) action pairs for router training.

        Three strategies to ensure pairs are generated even when all
        trajectories succeed on the first step:

        1. Intra-trajectory: compare steps with different score_deltas
        2. Cross-trajectory: compare successful steps from different
           domains/modes (the action that matched the domain is "good",
           the mismatched one is "bad")
        3. Synthetic negatives: for each successful step, generate a
           perturbed action embedding as the "bad" alternative
        """
        pairs = []
        _sq = lambda t: t.squeeze(0) if t.dim() > 1 else t

        # Collect all step records with their tensors
        all_steps = []
        for traj in self._buffer:
            tensors = self._tensor_store.get(traj.trajectory_id, {})
            for step in traj.steps:
                z_t = tensors.get(f"step_{step.step_idx}_z_t")
                action = tensors.get(f"step_{step.step_idx}_action_emb")
                z_hat = tensors.get(f"step_{step.step_idx}_z_hat")
                if z_t is not None and action is not None and z_hat is not None:
                    all_steps.append({
                        "step": step,
                        "domain": traj.domain,
                        "z_t": _sq(z_t),
                        "action": _sq(action),
                        "z_hat": _sq(z_hat),
                        "score": step.score_after,
                        "success": step.success,
                        "mode": step.mode,
                    })

        if len(all_steps) < 2:
            return pairs

        # Strategy 1: Intra-trajectory contrasts (original logic)
        for traj in self._buffer:
            tensors = self._tensor_store.get(traj.trajectory_id, {})
            steps_data = [s for s in all_steps if any(
                ts.trajectory_id == traj.trajectory_id for ts in self._buffer
                if any(st.step_idx == s["step"].step_idx for st in ts.steps)
            )]
            for i, si in enumerate(traj.steps):
                for sj in traj.steps:
                    if si.step_idx == sj.step_idx:
                        continue
                    ti = tensors.get(f"step_{si.step_idx}_z_t")
                    ai = tensors.get(f"step_{si.step_idx}_action_emb")
                    zi = tensors.get(f"step_{si.step_idx}_z_hat")
                    aj = tensors.get(f"step_{sj.step_idx}_action_emb")
                    zj = tensors.get(f"step_{sj.step_idx}_z_hat")
                    if any(x is None for x in [ti, ai, zi, aj, zj]):
                        continue
                    if si.score_delta > sj.score_delta:
                        pairs.append({
                            "z_t": _sq(ti), "action_good": _sq(ai),
                            "z_hat_good": _sq(zi), "action_bad": _sq(aj),
                            "z_hat_bad": _sq(zj),
                        })

        # Strategy 2: Cross-trajectory contrasts
        # A successful action on its own domain is "good"; that same state
        # paired with an action from a *different* domain is "bad"
        for i, si in enumerate(all_steps):
            if not si["success"]:
                continue
            for j, sj in enumerate(all_steps):
                if i == j:
                    continue
                # Different domain → the action from sj is likely worse for si's state
                if si["domain"] != sj["domain"]:
                    pairs.append({
                        "z_t": si["z_t"],
                        "action_good": si["action"],
                        "z_hat_good": si["z_hat"],
                        "action_bad": sj["action"],
                        "z_hat_bad": sj["z_hat"],
                    })
                    if len(pairs) > len(all_steps) * 4:
                        break
            if len(pairs) > len(all_steps) * 4:
                break

        # Strategy 3: Synthetic negatives — perturb successful actions
        for si in all_steps:
            if not si["success"]:
                continue
            noise = torch.randn_like(si["action"]) * 0.5
            bad_action = si["action"] + noise
            bad_z_hat = si["z_hat"] + torch.randn_like(si["z_hat"]) * 0.3
            pairs.append({
                "z_t": si["z_t"],
                "action_good": si["action"],
                "z_hat_good": si["z_hat"],
                "action_bad": bad_action,
                "z_hat_bad": bad_z_hat,
            })

        # Shuffle and cap
        random.shuffle(pairs)
        return pairs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str) -> None:
        """Save buffer to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata as JSON
        meta = [t.to_dict() for t in self._buffer]
        with open(path / "trajectories.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save tensors
        for tid, tensors in self._tensor_store.items():
            if tensors:
                torch.save(tensors, path / f"tensors_{tid}.pt")

    def load(self, directory: str) -> None:
        """Load buffer from disk."""
        path = Path(directory)
        if not (path / "trajectories.json").exists():
            return

        with open(path / "trajectories.json") as f:
            meta_list = json.load(f)

        for meta in meta_list:
            steps = [
                StepRecord(
                    step_idx=s["step_idx"],
                    mode=s["mode"],
                    budget=s["budget"],
                    strictness=s["strictness"],
                    tool_hint=s.get("tool_hint", ""),
                    success=s["success"],
                    score_before=s["score_before"],
                    score_after=s["score_after"],
                    elapsed_seconds=s.get("elapsed_seconds", 0),
                    solver_result=s.get("solver_result"),
                    verifier_result=s.get("verifier_result"),
                )
                for s in meta.get("steps", [])
            ]
            traj = TrajectoryRecord(
                trajectory_id=meta["trajectory_id"],
                task_summary=meta["task_summary"],
                domain=meta["domain"],
                steps=steps,
                final_success=meta["final_success"],
                final_score=meta["final_score"],
                total_elapsed=meta.get("total_elapsed", 0),
                timestamp=meta.get("timestamp", 0),
                metadata=meta.get("metadata", {}),
            )
            self._buffer.append(traj)

            # Load tensors if available
            tensor_path = path / f"tensors_{traj.trajectory_id}.pt"
            if tensor_path.exists():
                self._tensor_store[traj.trajectory_id] = torch.load(
                    tensor_path, map_location="cpu", weights_only=True
                )

    def statistics(self) -> Dict[str, Any]:
        """Compute summary statistics over the buffer."""
        if not self._buffer:
            return {"count": 0}

        successes = sum(1 for t in self._buffer if t.final_success)
        domains = {}
        mode_counts = {}
        for t in self._buffer:
            domains[t.domain] = domains.get(t.domain, 0) + 1
            for m in t.modes_used:
                mode_counts[m] = mode_counts.get(m, 0) + 1

        return {
            "count": len(self._buffer),
            "success_rate": successes / len(self._buffer),
            "avg_steps": sum(t.num_steps for t in self._buffer) / len(self._buffer),
            "avg_score": sum(t.final_score for t in self._buffer) / len(self._buffer),
            "domains": domains,
            "mode_counts": mode_counts,
        }
