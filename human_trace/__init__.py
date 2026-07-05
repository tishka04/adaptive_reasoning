"""Human-trace pipeline for the v4_1 adaptive reasoning system.

Black-box human play recorder + offline prior extraction for ARC-AGI-3.
See `human_trace/README.md` for the full design and usage.

Public entry points:
  - `human_trace.schema`       : step / episode dataclasses + JSONL I/O
  - `human_trace.recorder`     : Pygame-based human play wrapper
  - `human_trace.loader`       : load traces from disk
  - `human_trace.integration`  : seed v4_1 memories from traces
"""

from .schema import (
    CognitiveEvent,
    IntentTag,
    StepRecord,
    EpisodeRecord,
    TraceWriter,
    TraceReader,
)
from .loader import load_traces, TraceCorpus
from .integration import (
    HumanPriorPack,
    build_prior_pack,
    seed_game_memory,
    seed_cross_game_memory,
)
from .memory import HumanTraceMemory, HumanTrace, TraceSchema

__all__ = [
    "CognitiveEvent",
    "IntentTag",
    "StepRecord",
    "EpisodeRecord",
    "TraceWriter",
    "TraceReader",
    "load_traces",
    "TraceCorpus",
    "HumanPriorPack",
    "HumanTraceMemory",
    "HumanTrace",
    "TraceSchema",
    "build_prior_pack",
    "seed_game_memory",
    "seed_cross_game_memory",
]
