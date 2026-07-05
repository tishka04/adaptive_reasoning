"""
ARC-AGI-3 game adapter for the v4_1 adaptive reasoning system.

Bridges the interactive game environment with the adaptive reasoning
architecture: grid analysis, game-specific memory, and strategy routing.

Full reasoning loop:
  1. Observe  → StateDescriber
  2. Explore  → try actions, memorise
  3. Generate → StrategyGenerator (LLM / templates)
  4. Predict  → GameWorldModel (JEPA)
  5. Score    → GameEnergyScorer (EBM)
  6. Execute  → run winning strategy
  7. Learn    → update world model + EBM online
"""

from .grid_analyzer import GridAnalyzer, GridObject, FrameDiff
from .game_memory import GameMemory
from .state_describer import StateDescriber, GameObservation
from .strategy_generator import StrategyGenerator, GameStrategy, StrategyType
from .goal_decomposer import GoalDecomposer, GameGoal, SubGoal, SubGoalStatus
from .goal_pursuit import (
    GameObjective, ProgressSignal, ObjectiveStatus,
    StrategyOutcome, GoalProgressManager, GoalContext, TrajectoryOutcome,
)
from .actioner import Actioner, ActionResult
from .trajectory_memory import TrajectoryMemory

# Torch-dependent modules: import lazily to support environments without torch
try:
    from .trajectory_sampler import TrajectorySampler, SampledTrajectory
    from .game_world_model import GameWorldModel, WorldModelConfig
    from .energy_scorer import (
        GameEnergyScorer,
        ScoringDecision,
        TrajectoryScoringDecision,
    )
    from .reasoning_loop import AdaptiveReasoningLoop, LoopConfig, Phase
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
