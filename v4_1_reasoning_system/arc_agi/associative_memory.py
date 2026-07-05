"""
Brain-inspired associative memory for ARC-AGI-3 game agents.

Implements multiple memory systems analogous to biological memory:

  Episodic   — full action sequences per iteration with outcomes
  Semantic   — (state_feature, action) → strength associations (LTP/LTD)
  Procedural — winning action sequences that can be replayed
  Cross-game — meta-knowledge that transfers between games

Consolidation strengthens successful associations and weakens failures.
Retrieval uses cue-based pattern completion: current state → best action.
Forgetting prunes weak associations to prevent saturation.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import copy

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================
# Policy gradient network (actor-critic)
# ======================================================================

# Normalization constants for feature vector → float tensor
_FEAT_SCALE = torch.tensor([15.0, 10.0, 255.0, 77.0, 996.0, 1.0], dtype=torch.float32)


def _features_to_tensor(features: Tuple[int, ...]) -> torch.Tensor:
    """Convert discrete feature tuple to normalized [0,1] float tensor."""
    t = torch.tensor(features, dtype=torch.float32)
    return t / _FEAT_SCALE.clamp(min=1.0)


class PolicyValueNet(nn.Module):
    """Compact actor-critic: state features → (action logits, state value).

    Trained with REINFORCE + baseline (advantage actor-critic).
    Small enough to train online in milliseconds.
    """

    def __init__(self, n_features: int = 6, n_actions: int = 7, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class Episode:
    """One full game attempt (iteration)."""
    steps: List[EpisodeStep] = field(default_factory=list)
    max_level: int = 0
    total_reward: float = 0.0  # accumulated reward signal
    game_overs: int = 0

    @property
    def won(self) -> bool:
        return self.max_level > 0


@dataclass
class EpisodeStep:
    """A single (state, action, outcome) transition."""
    state_hash: int
    state_features: Tuple[int, ...]  # discretised feature vector
    action: int                       # action number (1-7)
    action_data: Optional[dict] = None
    changed: bool = False
    level_changed: bool = False
    game_over: bool = False
    reward: float = 0.0              # computed during consolidation


@dataclass
class Association:
    """A (state_feature, action) → strength mapping.

    Strength is analogous to synaptic weight.
    Positive = rewarding, negative = punishing.
    """
    strength: float = 0.0
    activations: int = 0
    last_activated: int = 0           # iteration number

    def potentiate(self, delta: float, iteration: int) -> None:
        """Long-Term Potentiation — strengthen the connection."""
        self.strength += delta
        self.activations += 1
        self.last_activated = iteration

    def depress(self, delta: float, iteration: int) -> None:
        """Long-Term Depression — weaken the connection."""
        self.strength -= delta
        self.activations += 1
        self.last_activated = iteration


# ======================================================================
# Feature extraction — turns raw grids into discrete cues
# ======================================================================

def extract_features(grid: Optional[np.ndarray], recent_actions: List[int]) -> Tuple[int, ...]:
    """Extract a compact, discretised feature vector from the current state.

    Features are designed to be coarse enough that similar states share
    features (enabling generalization) but specific enough to discriminate.
    """
    if grid is None:
        return (0, 0, 0, 0, 0, 0)

    h, w = grid.shape[:2]

    # F1: number of distinct non-zero values (object types)
    unique_vals = len(set(grid.flat) - {0})

    # F2: density bucket (what fraction of cells are non-zero)
    density = int(10 * np.count_nonzero(grid) / max(grid.size, 1))

    # F3: spatial hash — divide grid into 4x4 regions and hash occupancy
    rh, rw = max(h // 4, 1), max(w // 4, 1)
    spatial = 0
    for ry in range(4):
        for rx in range(4):
            region = grid[ry*rh:(ry+1)*rh, rx*rw:(rx+1)*rw]
            if region.any():
                spatial |= (1 << (ry * 4 + rx))
    spatial_bucket = spatial % 256  # 8-bit spatial fingerprint

    # F4: last 2 actions as a bigram (captures local context)
    bigram = 0
    if len(recent_actions) >= 2:
        bigram = recent_actions[-2] * 10 + recent_actions[-1]
    elif len(recent_actions) == 1:
        bigram = recent_actions[-1]

    # F5: compact grid hash (for exact state matching, mod small prime)
    grid_sig = hash(grid.tobytes()) % 997

    return (unique_vals, density, spatial_bucket, bigram, grid_sig, 0)


def feature_cues(features: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """Generate multiple cue keys from a feature vector.

    Each cue is a subset of the full feature vector.
    This enables partial-match retrieval (pattern completion).
    Weights indicate how much each cue type contributes.
    """
    f = features
    # (cue_tuple, weight)
    cues = [
        (f, 3.0),                                      # exact match (strongest)
        ((f[0], f[1], f[2], 0, f[4], 0), 2.0),        # state + grid_sig (no action context)
        ((f[0], f[1], f[2], f[3], 0, 0), 1.5),        # state + action context (no grid_sig)
        ((0, 0, f[2], f[3], 0, 0), 1.0),              # spatial + action context
        ((0, 0, 0, f[3], 0, 0), 0.5),                 # action context only
    ]
    return cues


# ======================================================================
# Associative Memory
# ======================================================================

class AssociativeMemory:
    """Multi-system associative memory inspired by biological memory.

    Usage:
        mem = AssociativeMemory()
        # During play:
        mem.begin_episode()
        mem.record_step(grid, action, action_data, changed, level_changed, game_over)
        mem.end_episode()
        # Consolidation happens automatically in end_episode().
        # To pick an action:
        action = mem.retrieve_action(grid, available_actions, recent_actions)
    """

    def __init__(
        self,
        ltp_rate: float = 0.3,      # learning rate for potentiation
        ltd_rate: float = 0.1,      # learning rate for depression
        decay_rate: float = 0.01,   # per-consolidation decay
        max_episodes: int = 200,    # cap before forgetting old episodes
        max_procedures: int = 20,   # top-K winning sequences
    ):
        # Semantic memory: (cue, action) → Association
        self.associations: Dict[Tuple[Tuple[int, ...], int], Association] = {}

        # Episodic memory: recent episodes
        self.episodes: List[Episode] = []
        self._current_episode: Optional[Episode] = None

        # Procedural memory: best winning sequences
        self.procedures: List[List[Tuple[int, Optional[dict]]]] = []

        # Dangerous state-action pairs (strong negative signal)
        self.danger_map: Dict[Tuple[Tuple[int, ...], int], float] = {}

        # Failed sequence trie — tracks action prefixes that didn't win.
        # Keys are tuple(action_sequence_so_far), values are set of
        # next-actions that have been tried and failed from that prefix.
        # This prevents repeating the same doomed paths.
        self._failed_prefixes: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
        self._max_prefix_len: int = 15   # only track first N actions

        # Per-game action effect tracking (model-based)
        # Much more sample-efficient than NN for 7 discrete actions.
        self._action_changes: Dict[int, int] = defaultdict(int)   # action → times it changed grid
        self._action_deaths: Dict[int, int] = defaultdict(int)    # action → times it caused game-over
        self._action_counts: Dict[int, int] = defaultdict(int)    # action → total times tried

        # Parameters
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.decay_rate = decay_rate
        self.max_episodes = max_episodes
        self.max_procedures = max_procedures

        # ── Policy gradient network (actor-critic) ────────────────
        self._policy_net = PolicyValueNet(n_features=6, n_actions=7, hidden=64)
        self._optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=3e-4)
        self._gamma: float = 0.99        # discount factor
        self._entropy_coef: float = 0.02 # exploration bonus
        self._value_coef: float = 0.5    # value loss weight
        self._nn_blend: float = 0.0      # how much NN influences action selection
        self._train_steps: int = 0
        self._replay_every: int = 5      # replay past episodes every N episodes

        # Visual cortex integration storage
        self._vc_change_prior: Dict[int, Tuple[int, int]] = {}  # act -> (count, changes)
        self._vc_directions: Dict[int, Tuple[float, float]] = {}  # act -> (dy, dx)
        self._vc_action_similarity: Dict[Tuple[int, int], float] = {}  # (a,b) -> cos

        # Statistics
        self.iteration: int = 0
        self.total_wins: int = 0
        self.total_game_overs: int = 0
        self._recent_actions: List[int] = []

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def begin_episode(self) -> None:
        """Start recording a new episode."""
        self._current_episode = Episode()
        self._recent_actions = []

    def record_step(
        self,
        grid: Optional[np.ndarray],
        action: int,
        action_data: Optional[dict] = None,
        changed: bool = False,
        level_changed: bool = False,
        game_over: bool = False,
    ) -> None:
        """Record one step in the current episode."""
        if self._current_episode is None:
            return

        features = extract_features(grid, self._recent_actions)
        state_hash = hash(grid.tobytes()) if grid is not None else 0

        step = EpisodeStep(
            state_hash=state_hash,
            state_features=features,
            action=action,
            action_data=action_data,
            changed=changed,
            level_changed=level_changed,
            game_over=game_over,
        )
        self._current_episode.steps.append(step)
        self._recent_actions.append(action)

        # Keep recent_actions window small
        if len(self._recent_actions) > 10:
            self._recent_actions = self._recent_actions[-10:]

        # Track per-action effects (model-based learning)
        if action != 0:
            self._action_counts[action] += 1
            if changed or level_changed:
                self._action_changes[action] += 1
            if game_over:
                self._action_deaths[action] += 1

        if level_changed:
            self._current_episode.max_level += 1
        if game_over:
            self._current_episode.game_overs += 1

    def end_episode(self) -> None:
        """End the current episode and run consolidation."""
        if self._current_episode is None:
            return

        ep = self._current_episode
        self.iteration += 1

        if ep.won:
            self.total_wins += 1
        self.total_game_overs += ep.game_overs

        # -- Assign rewards (hindsight) --
        self._assign_rewards(ep)

        # -- Consolidation (replay + LTP/LTD) --
        self._consolidate(ep)

        # -- Store episode --
        self.episodes.append(ep)

        # -- Store winning sequence in procedural memory --
        if ep.won:
            seq = [(s.action, s.action_data) for s in ep.steps]
            self.procedures.append(seq)
            self.procedures.sort(key=lambda s: len(s))  # prefer shorter
            if len(self.procedures) > self.max_procedures:
                self.procedures = self.procedures[:self.max_procedures]
        else:
            # -- Store failed sequence prefixes (avoidance learning) --
            self._record_failed_sequence(ep)

        # -- Train policy network on this episode --
        self._train_on_episode(ep)

        # -- Prioritized experience replay (train more on wins) --
        if self.iteration % self._replay_every == 0 and len(self.episodes) >= 3:
            self._replay_training()

        # -- Forgetting: prune old episodes --
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]

        # -- Synaptic decay (forgetting weak connections) --
        self._decay()

        self._current_episode = None

    # ------------------------------------------------------------------
    # Retrieval — cue-based pattern completion
    # ------------------------------------------------------------------
    def retrieve_action(
        self,
        grid: Optional[np.ndarray],
        available_actions: List[int],
        recent_actions: Optional[List[int]] = None,
        temperature: float = 1.0,
    ) -> Tuple[int, Optional[dict]]:
        """Retrieve the best action for the current state.

        Uses cue-based lookup into semantic associations.
        Falls back to random if no strong associations exist.

        Returns:
            (action_number, optional_action_data)
        """
        if recent_actions is None:
            recent_actions = self._recent_actions

        features = extract_features(grid, recent_actions)
        cues = feature_cues(features)

        # Score each available action
        scores: Dict[int, float] = defaultdict(float)
        for cue_tuple, cue_weight in cues:
            for act in available_actions:
                key = (cue_tuple, act)
                if key in self.associations:
                    assoc = self.associations[key]
                    # Weight by recency (more recent = stronger retrieval)
                    recency = 1.0 / (1.0 + 0.01 * (self.iteration - assoc.last_activated))
                    scores[act] += assoc.strength * recency * cue_weight

                # Check danger
                danger_key = (cue_tuple, act)
                if danger_key in self.danger_map:
                    scores[act] -= self.danger_map[danger_key] * cue_weight

        # Add exploration bonus for untried actions
        for act in available_actions:
            if scores[act] == 0:
                scores[act] += 0.1  # small bonus for novelty

        # Blend in policy network scores
        if self._nn_blend > 0 and self._train_steps > 0:
            feat_t = _features_to_tensor(features).unsqueeze(0)
            with torch.no_grad():
                logits, _ = self._policy_net(feat_t)
            # Mask unavailable actions
            mask = torch.full((7,), float('-inf'))
            for a in available_actions:
                if 1 <= a <= 7:
                    mask[a - 1] = 0.0
            nn_probs = F.softmax((logits[0] + mask) / max(temperature, 0.1), dim=-1)
            for a in available_actions:
                if 1 <= a <= 7:
                    scores[a] += self._nn_blend * nn_probs[a - 1].item()

        # Softmax selection with temperature
        if not scores:
            act = random.choice(available_actions)
            return act, None

        acts = list(scores.keys())
        vals = [scores[a] for a in acts]

        # Temperature-scaled softmax
        if temperature > 0:
            max_v = max(vals)
            exps = [math.exp((v - max_v) / max(temperature, 0.01)) for v in vals]
            total = sum(exps)
            probs = [e / total for e in exps]
            act = random.choices(acts, weights=probs, k=1)[0]
        else:
            act = acts[vals.index(max(vals))]

        # For ACTION6, try to retrieve a click position from procedural memory
        act_data = None
        if act == 6:
            act_data = self._retrieve_click_data(features)

        return act, act_data

    def _retrieve_click_data(self, features: Tuple[int, ...]) -> Optional[dict]:
        """Try to retrieve a good click position from past successful clicks."""
        # Search winning episodes for click data at similar states
        for ep in reversed(self.episodes):
            if not ep.won:
                continue
            for step in ep.steps:
                if step.action == 6 and step.action_data and step.changed:
                    return step.action_data
        return None

    # ------------------------------------------------------------------
    # Consolidation — replay + strengthen/weaken
    # ------------------------------------------------------------------
    def _assign_rewards(self, ep: Episode) -> None:
        """Assign reward to each step using hindsight (like TD(λ))."""
        n = len(ep.steps)
        if n == 0:
            return

        # Base reward: higher for winning episodes
        base = 1.0 if ep.won else 0.0

        for i, step in enumerate(ep.steps):
            r = 0.0

            # Positive signals
            if step.level_changed:
                r += 5.0               # strong: level completed
            elif step.changed:
                r += 0.2               # mild: something happened
            else:
                r -= 0.05              # small penalty: wasted action

            # Negative signals
            if step.game_over:
                r -= 3.0               # strong: died

            # Proximity bonus: steps near a level change get credit
            if ep.won:
                # Exponentially decayed credit from the winning step
                for j, s2 in enumerate(ep.steps):
                    if s2.level_changed:
                        dist = abs(i - j)
                        r += 2.0 * math.exp(-dist / 5.0)

            step.reward = r
            ep.total_reward += r

    def _consolidate(self, ep: Episode) -> None:
        """Replay the episode and update associations (LTP/LTD)."""
        for step in ep.steps:
            if step.action == 0:
                continue  # skip RESET markers

            cues = feature_cues(step.state_features)

            for cue_tuple, cue_weight in cues:
                key = (cue_tuple, step.action)

                if key not in self.associations:
                    self.associations[key] = Association()

                assoc = self.associations[key]

                if step.reward > 0:
                    # LTP: strengthen this association
                    assoc.potentiate(
                        self.ltp_rate * step.reward * cue_weight,
                        self.iteration,
                    )
                elif step.reward < 0:
                    # LTD: weaken this association
                    assoc.depress(
                        self.ltd_rate * abs(step.reward) * cue_weight,
                        self.iteration,
                    )

                # Update danger map for game-over actions
                if step.game_over:
                    danger_key = (cue_tuple, step.action)
                    self.danger_map[danger_key] = (
                        self.danger_map.get(danger_key, 0) + cue_weight
                    )

    def _decay(self) -> None:
        """Synaptic decay: weaken all associations slightly (forgetting).

        Removes very weak associations to prevent memory saturation.
        """
        to_remove = []
        for key, assoc in self.associations.items():
            # Time-based decay
            age = self.iteration - assoc.last_activated
            assoc.strength *= (1.0 - self.decay_rate)

            # Additional decay for old, weak connections
            if age > 20 and abs(assoc.strength) < 0.1:
                to_remove.append(key)

        for key in to_remove:
            del self.associations[key]

        # Also decay danger map slightly
        to_remove_danger = []
        for key in self.danger_map:
            self.danger_map[key] *= 0.99
            if self.danger_map[key] < 0.01:
                to_remove_danger.append(key)
        for key in to_remove_danger:
            del self.danger_map[key]

    # ------------------------------------------------------------------
    # Failed sequence tracking (avoidance learning)
    # ------------------------------------------------------------------
    def _record_failed_sequence(self, ep: Episode) -> None:
        """Record action prefixes from a failed episode.

        For each prefix length 1..N, note the next action chosen.
        Next time we reach the same prefix, we'll try a different action.
        """
        actions = [s.action for s in ep.steps if s.action != 0]
        limit = min(len(actions), self._max_prefix_len)
        for i in range(limit):
            prefix = tuple(actions[:i])
            next_act = actions[i]
            self._failed_prefixes[prefix].add(next_act)

    def pick_novel_action(
        self,
        available_actions: List[int],
        recent_sequence: List[int],
    ) -> int:
        """Pick an action biased toward productive ones, avoiding known failures.

        Uses per-game action effect stats to weight selection:
        - Actions that change the grid get higher weight
        - Actions that cause game-over get lower weight
        - Untried actions get moderate weight (encourage full coverage)
        - Known-failed sequence continuations are deprioritized
        """
        # Start with candidate set (avoidance learning)
        candidates = list(available_actions)
        prefix = tuple(recent_sequence[-self._max_prefix_len:])
        for plen in range(len(prefix), -1, -1):
            key = prefix[-plen:] if plen > 0 else ()
            if key in self._failed_prefixes:
                failed_acts = self._failed_prefixes[key]
                novel = [a for a in candidates if a not in failed_acts]
                if novel:
                    candidates = novel
                    break

        # Weight by action productivity (gentle bias — maintain diversity)
        weights = []
        for a in candidates:
            n = self._action_counts.get(a, 0)
            if n == 0:
                # Check if VC has a prior for this untried action
                vc_prior = self._vc_change_prior.get(a)
                if vc_prior is not None:
                    vc_n, vc_c = vc_prior
                    vc_rate = vc_c / max(vc_n, 1)
                    w = 0.5 + vc_rate * 1.0  # VC-informed weight
                else:
                    # Untried, no VC prior — high weight to encourage trying
                    w = 1.5
            else:
                change_rate = self._action_changes.get(a, 0) / n
                death_rate = self._action_deaths.get(a, 0) / n
                # Blend VC prior for actions with very few observations
                if n < 3 and a in self._vc_change_prior:
                    vc_n, vc_c = self._vc_change_prior[a]
                    vc_rate = vc_c / max(vc_n, 1)
                    # Weighted average: observed counts + VC virtual counts
                    change_rate = (self._action_changes.get(a, 0) + vc_c) / (n + vc_n)
                # Gentle bias: productive actions ~2-3x more likely, not 8x
                w = 0.3 + change_rate * 1.0 - death_rate * 0.5
                w = max(0.15, w)  # generous floor to keep diversity
            weights.append(w)

        return random.choices(candidates, weights=weights, k=1)[0]

    def clear_failed_prefixes(self) -> None:
        """Clear failed prefix memory (e.g. after a structural change)."""
        self._failed_prefixes.clear()

    # ------------------------------------------------------------------
    # Policy gradient training
    # ------------------------------------------------------------------
    def _compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ..."""
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self._gamma * G
            returns.insert(0, G)
        return returns

    def _train_on_episode(self, ep: Episode) -> None:
        """REINFORCE with baseline on one episode.

        Computes discounted returns, then updates the policy to increase
        probability of actions with positive advantage (return - baseline).
        """
        steps = [s for s in ep.steps if s.action != 0 and 1 <= s.action <= 7]
        if len(steps) < 2:
            return

        # Compute discounted returns from per-step rewards
        returns = self._compute_returns([s.reward for s in steps])

        # Build tensors
        feat_batch = torch.stack([_features_to_tensor(s.state_features) for s in steps])
        actions_t = torch.tensor([s.action - 1 for s in steps], dtype=torch.long)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns (reduces variance)
        if returns_t.std() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Forward pass
        logits, values = self._policy_net(feat_batch)

        # Advantages = returns - baseline (value estimate)
        advantages = returns_t - values.detach()

        # Policy loss: -log π(a|s) * A(s,a)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        policy_loss = -(action_log_probs * advantages).mean()

        # Value loss: MSE between V(s) and G_t
        value_loss = F.mse_loss(values, returns_t)

        # Entropy bonus: encourage exploration
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # VC similarity regularization: penalize divergent logits for
        # actions the visual cortex predicts have similar effects
        sim_loss = torch.tensor(0.0)
        if self._vc_action_similarity:
            mean_logits = logits.mean(dim=0)  # (7,) avg logits across batch
            for (a_i, a_j), cos_sim in self._vc_action_similarity.items():
                if cos_sim > 0.95 and 1 <= a_i <= 7 and 1 <= a_j <= 7:
                    # Very similar effects → logits should be close
                    sim_loss = sim_loss + (mean_logits[a_i - 1] - mean_logits[a_j - 1]) ** 2
            sim_loss = sim_loss * 0.01  # small coefficient — gentle nudge

        # Total loss
        loss = (policy_loss + self._value_coef * value_loss
                - self._entropy_coef * entropy + sim_loss)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy_net.parameters(), 1.0)
        self._optimizer.step()

        self._train_steps += 1
        # Gradually increase NN influence — cap low so tabular stays dominant
        self._nn_blend = min(0.5, self._train_steps * 0.002)

    def _replay_training(self) -> None:
        """Prioritized experience replay: sample past episodes and re-train.

        Winning episodes get 10× sampling weight — the network learns
        more from rare successes than frequent failures.
        """
        if len(self.episodes) < 3:
            return

        # Sampling weights: wins 10×, recent 2×
        weights = []
        n_eps = len(self.episodes)
        for i, ep in enumerate(self.episodes):
            w = 10.0 if ep.won else 1.0
            # Recency bonus
            w *= (1.0 + i / n_eps)
            weights.append(w)

        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        # Sample 3 episodes
        indices = random.choices(range(n_eps), weights=probs, k=min(3, n_eps))
        for idx in indices:
            self._train_on_episode(self.episodes[idx])

    # ------------------------------------------------------------------
    # Procedural memory — replay winning sequences
    # ------------------------------------------------------------------
    def get_best_procedure(self) -> Optional[List[Tuple[int, Optional[dict]]]]:
        """Return the shortest winning action sequence, if any."""
        if self.procedures:
            return self.procedures[0]
        return None

    # ------------------------------------------------------------------
    # Goal-indexed strategy outcomes
    # ------------------------------------------------------------------
    def record_strategy_outcome(self, outcome: 'StrategyOutcome') -> None:
        """Store a goal-conditioned strategy outcome for retrieval.

        Creates a retrieval axis indexed by goal_id, enabling:
          'what goal-conditioned strategy worked in situations like this?'
        """
        from .goal_pursuit import StrategyOutcome
        if not hasattr(self, '_strategy_outcomes'):
            self._strategy_outcomes: Dict[str, List] = {}

        goal_id = outcome.goal_id
        if goal_id not in self._strategy_outcomes:
            self._strategy_outcomes[goal_id] = []
        self._strategy_outcomes[goal_id].append(outcome)

        # Cap per-goal history to prevent memory bloat
        if len(self._strategy_outcomes[goal_id]) > 50:
            self._strategy_outcomes[goal_id] = self._strategy_outcomes[goal_id][-30:]

    def retrieve_goal_strategies(
        self, goal_id: str, min_progress: float = 0.0
    ) -> List['StrategyOutcome']:
        """Retrieve past strategy outcomes for a given goal.

        Useful for: showing the strategy generator what already worked/failed.
        """
        if not hasattr(self, '_strategy_outcomes'):
            return []
        outcomes = self._strategy_outcomes.get(goal_id, [])
        if min_progress > 0:
            outcomes = [o for o in outcomes if o.progress_score >= min_progress]
        return outcomes

    def get_best_goal_strategy(
        self, goal_id: str
    ) -> Optional['StrategyOutcome']:
        """Return the highest-progress strategy outcome for a goal."""
        outcomes = self.retrieve_goal_strategies(goal_id)
        if not outcomes:
            return None
        return max(outcomes, key=lambda o: o.progress_score)

    # ------------------------------------------------------------------
    # Cross-game learning
    # ------------------------------------------------------------------
    def new_game(self, cross_game: Optional['CrossGameMemory'] = None) -> None:
        """Reset per-game state but inherit cross-game meta-knowledge.

        Call this when starting a new game. Preserves:
          - NN policy weights (generalised exploration strategy)
          - Action semantic priors from previous games
          - Game-type classification priors
        Resets:
          - Associations, episodes, procedures (game-specific)
          - Failed prefixes, action effect counts (game-specific)
          - Danger map, recent actions
        """
        # Reset per-game state
        self.associations.clear()
        self.episodes.clear()
        self._current_episode = None
        self.procedures.clear()
        self.danger_map.clear()
        self._failed_prefixes.clear()
        self._action_changes.clear()
        self._action_deaths.clear()
        self._action_counts.clear()
        self.iteration = 0
        self.total_wins = 0
        self.total_game_overs = 0
        self._recent_actions = []
        self._vc_change_prior.clear()
        self._vc_directions.clear()
        self._vc_action_similarity.clear()
        if hasattr(self, '_strategy_outcomes'):
            self._strategy_outcomes.clear()

        # Inherit cross-game priors (if available)
        # IMPORTANT: priors are HYPOTHESES, not facts. They are gated by
        # trust_factor (0.3 dev / 0.15 competition) and must be validated
        # by in-game evidence before gaining influence.
        if cross_game is not None:
            trust = cross_game.prior_trust

            # Transfer NN weights — but scale blend by trust so the NN
            # starts as a weak suggestion, not an action governor
            self._policy_net.load_state_dict(
                copy.deepcopy(cross_game.policy_net_state)
            )
            self._optimizer = torch.optim.Adam(
                self._policy_net.parameters(), lr=3e-4
            )
            self._train_steps = cross_game.total_train_steps
            # NN blend gated by trust: starts low, grows as in-game
            # training confirms the policy is useful for THIS game
            raw_blend = min(0.5, self._train_steps * 0.002)
            self._nn_blend = raw_blend * trust

            # Seed action effect priors — reduced by trust factor so
            # in-game observations quickly dominate
            for act, prior in cross_game.action_priors.items():
                scaled_count = max(1, int(prior["count_prior"] * trust))
                self._action_counts[act] = scaled_count
                self._action_changes[act] = int(prior["change_prior"] * trust)
                self._action_deaths[act] = int(prior["death_prior"] * trust)

            # Cross-game hints are read-only hypotheses (lightweight dicts),
            # stored separately from in-game StrategyOutcome objects to
            # avoid type-mixing and memory bloat.
            self._cross_game_hints = dict(cross_game.goal_strategy_hints)

            # Store failure memory reference so in-game code can query it
            self._cross_game_failures = cross_game.failure_patterns
            self._cross_game_overpredicted = cross_game.overpredicted_goals

    def export_to_cross_game(self, cross_game: 'CrossGameMemory') -> None:
        """Export this game's learnings back to cross-game memory.

        Persists both successes AND failures. Detects proxy-progress
        patterns (high progress score but no actual win) and records
        them as anti-hints so future games can be sceptical.
        """
        # Update NN weights
        cross_game.policy_net_state = copy.deepcopy(
            self._policy_net.state_dict()
        )
        cross_game.total_train_steps = self._train_steps

        won_this_game = self.total_wins > 0

        # Export strategy outcomes — BOTH successes and failures
        # IMPORTANT: store only lightweight dicts, never full StrategyOutcome
        # objects (which carry action_sequence lists that cause memory blowup).
        if hasattr(self, '_strategy_outcomes') and hasattr(cross_game, 'goal_strategy_hints'):
            from .goal_pursuit import PARTIAL_THRESHOLD

            # Normalise goal IDs to prevent key proliferation from LLM
            _CANONICAL_GOALS = {
                "click_puzzle", "sequence_puzzle", "navigate_exit",
                "navigate_avoid_hazards", "push_puzzle",
                "discover_mechanics", "collect",
            }
            def _normalise_gid(raw_gid: str) -> str:
                low = raw_gid.lower().replace(" ", "_")
                for canon in _CANONICAL_GOALS:
                    if canon in low:
                        return canon
                if low.startswith("navigate_to_") or "navigate" in low:
                    return "navigate_exit"
                if low.startswith("collect"):
                    return "collect"
                return "discover_mechanics"  # fallback bucket

            for gid, outcomes in self._strategy_outcomes.items():
                norm_gid = _normalise_gid(gid)
                good = [o for o in outcomes if o.progress_score >= PARTIAL_THRESHOLD]
                if good:
                    if norm_gid not in cross_game.goal_strategy_hints:
                        cross_game.goal_strategy_hints[norm_gid] = []
                    # Convert to lightweight dicts (no action_sequence)
                    for o in good:
                        cross_game.goal_strategy_hints[norm_gid].append(
                            o.to_dict() if hasattr(o, 'to_dict') else {
                                "goal_id": norm_gid,
                                "strategy": str(o.strategy_description)[:80],
                                "progress": o.progress_score,
                                "status": o.terminal_status,
                            }
                        )
                    # Keep only top-3 per normalised goal type
                    cross_game.goal_strategy_hints[norm_gid].sort(
                        key=lambda x: -(x.get('progress', 0) if isinstance(x, dict)
                                        else getattr(x, 'progress_score', 0))
                    )
                    cross_game.goal_strategy_hints[norm_gid] = \
                        cross_game.goal_strategy_hints[norm_gid][:3]

                # ── Failure persistence ──
                # If this goal had progress but no win → record as
                # potential proxy-progress pattern
                best_progress = max((o.progress_score for o in outcomes), default=0)
                if best_progress >= PARTIAL_THRESHOLD and not won_this_game:
                    if norm_gid not in cross_game.failure_patterns:
                        cross_game.failure_patterns[norm_gid] = []
                    # Keep last 5 failure records per normalised goal type
                    cross_game.failure_patterns[norm_gid].append({
                        "strategy": str(outcomes[0].strategy_description)[:80] if outcomes else "",
                        "reason": "proxy_progress_no_win",
                        "proxy_progress": round(best_progress, 3),
                        "actual_win": False,
                    })
                    cross_game.failure_patterns[norm_gid] = \
                        cross_game.failure_patterns[norm_gid][-5:]

                    # Track over-predicted goal families
                    cross_game.overpredicted_goals[norm_gid] = \
                        cross_game.overpredicted_goals.get(norm_gid, 0) + 1

        # Update action priors (running average across games)
        # Also detect contradictions: if this game's observation strongly
        # disagrees with the prior, record it.
        for act in self._action_counts:
            count = self._action_counts[act]
            if count == 0:
                continue
            change_rate = self._action_changes.get(act, 0) / count
            death_rate = self._action_deaths.get(act, 0) / count

            existing = cross_game.action_priors.get(act, {
                "change_rate": 0.5, "death_rate": 0.1,
                "count_prior": 1, "change_prior": 0, "death_prior": 0,
                "games_seen": 0,
            })

            # Detect contradiction: this game disagrees with prior by > 0.3
            if existing["games_seen"] > 0:
                cr_diff = abs(change_rate - existing["change_rate"])
                dr_diff = abs(death_rate - existing["death_rate"])
                if cr_diff > 0.3 or dr_diff > 0.3:
                    cross_game.contradicted_priors[act] = \
                        cross_game.contradicted_priors.get(act, 0) + 1
                    logger.debug(
                        f"Action {act} prior contradicted: "
                        f"prior_cr={existing['change_rate']:.2f} vs obs_cr={change_rate:.2f}"
                    )

            n = existing["games_seen"] + 1
            alpha = 1.0 / n
            existing["change_rate"] = (1 - alpha) * existing["change_rate"] + alpha * change_rate
            existing["death_rate"] = (1 - alpha) * existing["death_rate"] + alpha * death_rate
            existing["games_seen"] = n
            # Prior counts for next game (small — just a hint, not dominating)
            # Reduce count for frequently-contradicted actions
            contradiction_penalty = cross_game.contradicted_priors.get(act, 0)
            max_count = max(3, 10 - contradiction_penalty)
            existing["count_prior"] = max(1, min(max_count, n))
            existing["change_prior"] = int(existing["change_rate"] * existing["count_prior"])
            existing["death_prior"] = int(existing["death_rate"] * existing["count_prior"])
            cross_game.action_priors[act] = existing

        # Record game outcome (diagnostics only — see class docstring)
        cross_game.games_played += 1
        if won_this_game:
            cross_game.games_won += 1

    # ------------------------------------------------------------------
    # Visual cortex integration
    # ------------------------------------------------------------------
    def ingest_vc_predictions(
        self,
        vc_analysis: Dict[int, Dict[str, Any]],
        vc_similarity: Dict[Tuple[int, int], float],
        grid: Optional[np.ndarray] = None,
    ) -> None:
        """Ingest structured predictions from the visual cortex.

        Calls all four integration pathways:
          1. Predicted change rates -> action weights
          2. Predicted directions -> direction knowledge
          3. Predicted danger -> danger_map
          4. Action similarity -> policy NN generalization

        Args:
            vc_analysis: from VisualCortex.analyze_action_effects()
            vc_similarity: from VisualCortex.compute_action_similarity()
            grid: current grid (needed for danger_map state hashing)
        """
        if not vc_analysis:
            return
        self._ingest_vc_change_rates(vc_analysis)
        self._ingest_vc_directions(vc_analysis)
        self._ingest_vc_danger(vc_analysis, grid)
        self._ingest_vc_similarity(vc_similarity)

    def _ingest_vc_change_rates(
        self, vc_analysis: Dict[int, Dict[str, Any]],
    ) -> None:
        """(1) Predicted change rates bias pick_novel_action weights.

        VC-predicted change rates are blended into the action effect
        counters as "virtual observations".  Uses a small count (2)
        so real observations quickly overwhelm the prior.
        """
        for act, info in vc_analysis.items():
            cr = info.get("change_rate", 0.0)
            if act not in self._action_counts or self._action_counts[act] < 3:
                # Only inject prior when we have few real observations
                vc_count = 2
                vc_changes = int(round(cr * vc_count))
                self._vc_change_prior[act] = (vc_count, vc_changes)

    def _ingest_vc_directions(
        self, vc_analysis: Dict[int, Dict[str, Any]],
    ) -> None:
        """(2) Predicted movement directions -> _vc_directions.

        Stored separately so the actioner can use them for navigation
        when GameMemory doesn't have observed directions yet.
        """
        for act, info in vc_analysis.items():
            direction = info.get("direction")
            if direction is not None:
                self._vc_directions[act] = direction

    def _ingest_vc_danger(
        self,
        vc_analysis: Dict[int, Dict[str, Any]],
        grid: Optional[np.ndarray],
    ) -> None:
        """(3) High danger scores -> danger_map entries.

        If the VC predicts a drastic change (>30% cells or many
        disappearances), inject a danger signal into the danger_map
        at the current state hash.
        """
        if grid is None:
            return
        features = extract_features(grid, self._recent_actions)
        cues = feature_cues(features)

        for act, info in vc_analysis.items():
            danger = info.get("danger_score", 0.0)
            if danger > 0.3:
                for cue_tuple, cue_weight in cues:
                    key = (cue_tuple, act)
                    # Blend with existing danger (don't overwrite observed data)
                    existing = self.danger_map.get(key, 0.0)
                    # VC danger is a soft hint — cap at 0.5 to not dominate observed deaths
                    vc_contribution = min(danger * 0.5, 0.5)
                    self.danger_map[key] = max(existing, vc_contribution * cue_weight)

    def _ingest_vc_similarity(
        self,
        vc_similarity: Dict[Tuple[int, int], float],
    ) -> None:
        """(4) Action similarity -> policy NN generalization.

        When actions have very similar predicted effects (cosine > 0.95),
        their policy logits should be similar too.  We store the similarity
        matrix so the NN's training can penalize divergent logits for
        similar-effect actions.
        """
        self._vc_action_similarity = dict(vc_similarity)

    def get_vc_direction(self, action: int) -> Optional[Tuple[float, float]]:
        """Get VC-predicted movement direction for an action."""
        return self._vc_directions.get(action)

    def stats(self) -> Dict[str, Any]:
        """Summary statistics."""
        return {
            "iterations": self.iteration,
            "total_wins": self.total_wins,
            "total_game_overs": self.total_game_overs,
            "associations": len(self.associations),
            "danger_entries": len(self.danger_map),
            "procedures": len(self.procedures),
            "episodes_stored": len(self.episodes),
            "failed_prefixes": len(self._failed_prefixes),
            "nn_train_steps": self._train_steps,
            "nn_blend": round(self._nn_blend, 2),
            "action_effects": {
                a: f"{self._action_changes.get(a,0)}/{self._action_counts.get(a,0)}"
                for a in sorted(self._action_counts.keys())
            },
            "win_rate": self.total_wins / max(self.iteration, 1),
        }


# ======================================================================
# Cross-game memory — persists across games
# ======================================================================

class CrossGameMemory:
    """Meta-knowledge that transfers between games.

    Stores:
      - Policy NN weights (generalised exploration strategy)
      - Action effect priors (average change/death rates across games)
      - Game outcome history (for tracking progress)

    Usage::

        xmem = CrossGameMemory()
        for game_id in games:
            brain = AssociativeMemory(...)
            brain.new_game(xmem)          # inherit priors
            # ... play game ...
            brain.export_to_cross_game(xmem)  # save learnings
    """

    # ── Mode constants ─────────────────────────────────────────
    MODE_DEVELOPMENT = "development"    # full persistence on known games
    MODE_COMPETITION = "competition"    # heavy regularisation for novelty

    # Trust parameters: cross-run priors are *hypotheses*, not facts
    INITIAL_TRUST = 0.3        # how much to trust persistent priors at game start
    TRUST_DECAY_ON_DISAGREE = 0.5   # halve trust when current game contradicts
    TRUST_GROWTH_ON_AGREE = 0.1     # slow climb when current game confirms
    MAX_TRUST = 0.7            # priors never dominate in-game evidence

    def __init__(self, mode: str = "development") -> None:
        self.mode = mode

        # NN policy weights — initialised from a fresh network
        _net = PolicyValueNet(n_features=6, n_actions=7, hidden=64)
        self.policy_net_state: Dict[str, Any] = copy.deepcopy(_net.state_dict())
        self.total_train_steps: int = 0

        # Action effect priors (per action int)
        # Each entry: {change_rate, death_rate, count_prior, change_prior,
        #              death_prior, games_seen}
        self.action_priors: Dict[int, Dict[str, Any]] = {}

        # Goal-indexed strategy outcome hints (transferred across games)
        # Each entry: goal_id → list of StrategyOutcome (top-3 by progress)
        self.goal_strategy_hints: Dict[str, List] = {}
        # Goal-family-indexed compact trajectory fragments for proposal seeding.
        self.trajectory_priors: Dict[str, List[Dict[str, Any]]] = {}

        # ── Failure memory (as important as success memory) ──────
        # goal_type → list of {strategy, reason, proxy_progress, actual_wins}
        self.failure_patterns: Dict[str, List[Dict]] = {}
        # goal families that are frequently over-predicted by LLM
        self.overpredicted_goals: Dict[str, int] = {}  # goal_type → count
        # action priors that were contradicted in-game
        self.contradicted_priors: Dict[int, int] = {}  # action → contradiction count

        # ── Diagnostics only — NEVER used in control logic ──────
        # These counters exist purely for reporting. No method other
        # than stats() and save/load may read them.
        self.games_played: int = 0
        self.games_won: int = 0

    @property
    def prior_trust(self) -> float:
        """Current trust level in persistent priors (for new_game seeding)."""
        if self.mode == self.MODE_COMPETITION:
            return self.INITIAL_TRUST * 0.5  # extra sceptical in competition
        return self.INITIAL_TRUST

    def stats(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "games_played": self.games_played,
            "games_won": self.games_won,
            "win_rate": self.games_won / max(self.games_played, 1),
            "nn_train_steps": self.total_train_steps,
            "action_priors": {
                a: f"chg={p['change_rate']:.0%} die={p['death_rate']:.0%} n={p['games_seen']}"
                for a, p in sorted(self.action_priors.items())
            },
            "failure_patterns": len(self.failure_patterns),
            "trajectory_priors": {
                gid: len(items) for gid, items in self.trajectory_priors.items()
            },
            "overpredicted_goals": dict(self.overpredicted_goals),
            "contradicted_priors": dict(self.contradicted_priors),
        }

    # ── Persistence ──────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Save cross-game memory to disk so it persists between runs."""
        import torch
        data = {
            "policy_net_state": self.policy_net_state,
            "total_train_steps": self.total_train_steps,
            "action_priors": self.action_priors,
            "goal_strategy_hints": {
                k: [
                    o if isinstance(o, dict)
                    else (o.to_dict() if hasattr(o, "to_dict") else {"strategy": str(o)[:80]})
                    for o in v
                ]
                for k, v in self.goal_strategy_hints.items()
            },
            "trajectory_priors": {
                k: [dict(item) for item in v if isinstance(item, dict)]
                for k, v in self.trajectory_priors.items()
            },
            "failure_patterns": self.failure_patterns,
            "overpredicted_goals": dict(self.overpredicted_goals),
            "contradicted_priors": dict(self.contradicted_priors),
            "mode": self.mode,
            "games_played": self.games_played,
            "games_won": self.games_won,
        }
        torch.save(data, path)
        # Size guard: warn if file is suspiciously large
        import os
        fsize = os.path.getsize(path)
        if fsize > 10 * 1024 * 1024:  # 10 MB
            logger.warning(
                f"CrossGameMemory file is {fsize / 1024 / 1024:.1f} MB — "
                f"possible memory leak! Hints: {len(self.goal_strategy_hints)} types, "
                f"failures: {sum(len(v) for v in self.failure_patterns.values())} records"
            )
        logger.info(f"CrossGameMemory saved to {path} ({self.games_played} games, {fsize/1024:.0f} KB)")

    @classmethod
    def load(cls, path: str) -> "CrossGameMemory":
        """Load cross-game memory from disk. Returns fresh instance if file missing."""
        import torch, os
        if not os.path.exists(path):
            logger.info(f"No cross-game memory at {path}, starting fresh")
            return cls()
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            obj = cls()
            obj.policy_net_state = data["policy_net_state"]
            obj.total_train_steps = data.get("total_train_steps", 0)
            obj.action_priors = data.get("action_priors", {})
            obj.games_played = data.get("games_played", 0)
            obj.games_won = data.get("games_won", 0)
            obj.failure_patterns = data.get("failure_patterns", {})
            obj.trajectory_priors = {
                gid: [dict(item) for item in items if isinstance(item, dict)]
                for gid, items in (data.get("trajectory_priors", {}) or {}).items()
            }
            obj.overpredicted_goals = data.get("overpredicted_goals", {})
            obj.contradicted_priors = data.get("contradicted_priors", {})
            obj.mode = data.get("mode", cls.MODE_DEVELOPMENT)
            # Restore goal strategy hints as lightweight dicts (never full objects)
            raw_hints = data.get("goal_strategy_hints", {})
            for gid, hint_list in raw_hints.items():
                cleaned = []
                for h in hint_list:
                    if isinstance(h, dict):
                        # Keep only the fields we need (strip any junk)
                        cleaned.append({
                            "goal_id": h.get("goal_id", gid),
                            "strategy": str(h.get("strategy", h.get("goal_desc", "")))[:80],
                            "strategy_type": h.get("strategy_type", ""),
                            "progress": h.get("progress", 0.0),
                            "status": h.get("status", "unknown"),
                            "actions": h.get("actions", 0),
                            "time": h.get("time", 0.0),
                        })
                    elif hasattr(h, "to_dict"):
                        # Legacy: convert StrategyOutcome object to dict
                        cleaned.append(h.to_dict())
                if cleaned:
                    obj.goal_strategy_hints[gid] = cleaned[:3]  # cap at 3
            logger.info(
                f"CrossGameMemory loaded from {path}: "
                f"{obj.games_played} games, {obj.total_train_steps} NN steps"
            )
            return obj
        except Exception as e:
            logger.warning(f"Failed to load cross-game memory: {e}, starting fresh")
            return cls()
