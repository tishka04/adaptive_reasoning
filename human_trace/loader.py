"""Load human traces from disk into an in-memory corpus."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .schema import EpisodeRecord, StepRecord, TraceReader


@dataclass
class TraceCorpus:
    """All traces grouped by game_id → list of episodes.

    Each episode entry is `{"episode": EpisodeRecord | None, "steps": [StepRecord, ...]}`.
    An episode with `episode = None` means the steps were written but the
    session was killed before an EpisodeRecord could be emitted (rare).
    """
    by_game: Dict[str, List[Dict[str, object]]] = field(default_factory=dict)

    @property
    def n_episodes(self) -> int:
        return sum(len(v) for v in self.by_game.values())

    @property
    def n_steps(self) -> int:
        return sum(
            len(ep["steps"])  # type: ignore[arg-type]
            for eps in self.by_game.values()
            for ep in eps
        )


def load_traces(trace_dir: str | Path, game_id: Optional[str] = None) -> TraceCorpus:
    """Scan `trace_dir` for JSONL trace files and group them.

    If `game_id` is given, only files whose name starts with that prefix are loaded.
    """
    reader = TraceReader(trace_dir)
    grouped = reader.group_by_episode(game_id=game_id)

    corpus = TraceCorpus()
    for ep_id, bucket in grouped.items():
        steps: List[StepRecord] = bucket["steps"]  # type: ignore[assignment]
        ep: Optional[EpisodeRecord] = bucket["episode"]  # type: ignore[assignment]
        if not steps and ep is None:
            continue
        gid = ep.game_id if ep else (steps[0].game_id if steps else "unknown")
        corpus.by_game.setdefault(gid, []).append({"episode": ep, "steps": steps})

    return corpus
