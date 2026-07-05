"""Composition chamber for V4."""

from .forgetting import ForgettingManager
from .motif_composer import MotifComposer
from .prefix_compressor import PrefixCompressor
from .ritualizer import Ritualizer

__all__ = [
    "ForgettingManager",
    "MotifComposer",
    "PrefixCompressor",
    "Ritualizer",
]
