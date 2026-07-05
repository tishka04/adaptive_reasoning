"""Schema for human-play traces.

Two layers, as described in `V5_PROPOSAL.md` / the user's spec:

1. `StepRecord` — one black-box transition annotated with a single intent
   tag, optional cognitive event markers, and optionally a sticky hypothesis
   string.
2. `EpisodeRecord` — one full attempt (from RESET to terminal/quit) with
   outcome, game-type guess, objective guess, and discovered mechanics.

Traces are written as JSONL:
  - `<game_id>.<YYYYMMDD-HHMMSS>.steps.jsonl`   (one StepRecord per line)
  - `<game_id>.<YYYYMMDD-HHMMSS>.episodes.jsonl`(one EpisodeRecord per line)

Both files are produced by `TraceWriter` during recording and consumed by
`TraceReader` during offline prior extraction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


class IntentTag(str, Enum):
    """Short cognitive tag the human attaches to each action.

    Kept deliberately small so annotation stays cheap. Matches the set
    proposed in the user's original spec.
    """
    TEST_MOVE = "test_move"
    TEST_CLICK = "test_click"
    TEST_INTERACTION = "test_interaction"
    PROBE_OBJECT = "probe_object"
    AVOID_DANGER = "avoid_danger"
    REACH_TARGET = "reach_target"
    EXPLORE_UNKNOWN = "explore_unknown"
    REPEAT_SUCCESS = "repeat_success"
    RECOVER_AFTER_FAILURE = "recover_after_failure"
    NONE = "none"

    @classmethod
    def cycle_order(cls) -> List["IntentTag"]:
        """Order used by the recorder's Tab-cycle UI."""
        return [
            cls.EXPLORE_UNKNOWN,
            cls.TEST_MOVE,
            cls.TEST_CLICK,
            cls.TEST_INTERACTION,
            cls.PROBE_OBJECT,
            cls.REACH_TARGET,
            cls.AVOID_DANGER,
            cls.REPEAT_SUCCESS,
            cls.RECOVER_AFTER_FAILURE,
            cls.NONE,
        ]


class CognitiveEvent(str, Enum):
    """Rare step-level markers for cognitive turning points.

    These are not phases. They mark events the human believes happened in the
    current state before the next recorded action.
    """

    HYPOTHESIS_CONFIRMED = "hypothesis_confirmed"
    HYPOTHESIS_REJECTED = "hypothesis_rejected"
    GOAL_CHANGED = "goal_changed"

    @classmethod
    def labels(cls) -> List[str]:
        return [event.value for event in cls]


# ------------------------------------------------------------------
# Step-level record
# ------------------------------------------------------------------

@dataclass
class StepRecord:
    """One black-box transition: (s_t, a_t, s_{t+1}) with human annotation.

    Frames are stored as the primary grid (last layer of `FrameData.frame`)
    to keep trace files small. Full multi-layer frames are not required
    for seeding v4_1 priors.
    """
    game_id: str
    episode_id: str
    step: int
    frame_before: List[List[int]]          # 2D grid (primary layer)
    available_actions: List[int]           # GameAction ids
    action: str                            # GameAction name e.g. "ACTION6"
    action_args: Optional[Dict[str, Any]]  # {"x": int, "y": int} for ACTION6, else None
    frame_after: List[List[int]]
    game_state_after: str                  # "NOT_FINISHED" / "WIN" / "GAME_OVER"
    levels_completed_after: int
    intent: str                            # IntentTag value
    cognitive_events: List[str] = field(default_factory=list)
    hypothesis: str = ""                   # sticky hypothesis, may be empty
    t_ms: int = 0                          # ms since episode start, for debugging

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "StepRecord":
        d = json.loads(line)
        d.setdefault("cognitive_events", [])
        return cls(**d)


# ------------------------------------------------------------------
# Episode-level record
# ------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    """Aggregated record of one playthrough."""
    game_id: str
    episode_id: str
    started_at: str                    # ISO UTC
    ended_at: str                      # ISO UTC
    n_steps: int
    final_state: str                   # "WIN" / "GAME_OVER" / "QUIT"
    levels_completed: int
    game_type_guess: str = ""          # human's best guess: "click_puzzle", etc.
    objective_guess: str = ""          # short NL: "activate all targets"
    discovered_mechanics: List[str] = field(default_factory=list)
    discovered_mistakes: List[str] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "EpisodeRecord":
        d = json.loads(line)
        return cls(**d)


# ------------------------------------------------------------------
# JSONL writer / reader
# ------------------------------------------------------------------

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


class TraceWriter:
    """Append-only JSONL writer for human traces.

    Creates two files under `out_dir`:
        <game_id>.<stamp>.steps.jsonl
        <game_id>.<stamp>.episodes.jsonl
    """

    def __init__(self, out_dir: str | Path, game_id: str, stamp: Optional[str] = None) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.game_id = game_id
        self.stamp = stamp or _utc_stamp()
        self.steps_path = self.out_dir / f"{game_id}.{self.stamp}.steps.jsonl"
        self.episodes_path = self.out_dir / f"{game_id}.{self.stamp}.episodes.jsonl"
        # Touch the files so they exist even for empty sessions.
        self.steps_path.touch(exist_ok=True)
        self.episodes_path.touch(exist_ok=True)

    def write_step(self, step: StepRecord) -> None:
        with self.steps_path.open("a", encoding="utf-8") as f:
            f.write(step.to_json() + "\n")

    def write_episode(self, ep: EpisodeRecord) -> None:
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(ep.to_json() + "\n")


class TraceReader:
    """Iterate over traces on disk. Pairs steps/episodes by (game_id, episode_id)."""

    def __init__(self, in_dir: str | Path) -> None:
        self.in_dir = Path(in_dir)

    def iter_steps(self, game_id: Optional[str] = None) -> Iterator[StepRecord]:
        for p in sorted(self.in_dir.glob("*.steps.jsonl")):
            if game_id and not p.name.startswith(f"{game_id}."):
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield StepRecord.from_json(line)
                    except (json.JSONDecodeError, TypeError):
                        continue

    def iter_episodes(self, game_id: Optional[str] = None) -> Iterator[EpisodeRecord]:
        for p in sorted(self.in_dir.glob("*.episodes.jsonl")):
            if game_id and not p.name.startswith(f"{game_id}."):
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield EpisodeRecord.from_json(line)
                    except (json.JSONDecodeError, TypeError):
                        continue

    def group_by_episode(
        self, game_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Return {episode_id: {"episode": EpisodeRecord|None, "steps": [StepRecord,...]}}."""
        out: Dict[str, Dict[str, Any]] = {}
        for s in self.iter_steps(game_id=game_id):
            out.setdefault(s.episode_id, {"episode": None, "steps": []})["steps"].append(s)
        for e in self.iter_episodes(game_id=game_id):
            bucket = out.setdefault(e.episode_id, {"episode": None, "steps": []})
            bucket["episode"] = e
        return out
