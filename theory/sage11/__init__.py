"""SAGE.11 neuro-symbolic world-model and authority stack.

The package is intentionally separated from the historical M2/v4 models.  Its
split firewall, dataset artifacts, checkpoints, and runtime authority modes are
versioned independently so no legacy weight can silently enter an experiment.
"""

from .authority import (
    NeuralActionPrediction,
    NeuralAuthorityConfig,
    NeuralAuthorityMode,
    NeuroSymbolicRanker,
)
from .splits import (
    HISTORICAL_BENCHMARK,
    NEURO_HOLDOUT_V1,
    SOURCE_TRAIN,
    SOURCE_VALIDATION,
    SAGE11_SPLITS,
)

__all__ = [
    "HISTORICAL_BENCHMARK",
    "NEURO_HOLDOUT_V1",
    "NeuralActionPrediction",
    "NeuralAuthorityConfig",
    "NeuralAuthorityMode",
    "NeuroSymbolicRanker",
    "SAGE11_SPLITS",
    "SOURCE_TRAIN",
    "SOURCE_VALIDATION",
]
