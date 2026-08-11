"""SAGE.T: a unified posterior over complete executable world programs."""

from .compiler import compile_observation, compile_transition_record
from .consolidation import (
    ConsolidationEntry,
    ConsolidationRegistry,
    LegacyArbiter,
)
from .contracts import (
    AbstractEntity,
    AbstractState,
    ActionBinding,
    ActionCandidate,
    Effect,
    Expression,
    GoalRule,
    GroundFact,
    JointProgramHypothesis,
    ObjectSchema,
    ObservedTransition,
    PredictionPacket,
    ProgramFragment,
    ProgressRule,
    TerminalRule,
    TransitionRule,
    TruthValue,
    program_from_dict,
    program_to_dict,
)
from .controller import (
    SageTArbitration,
    SageTConfig,
    SageTController,
    SageTMode,
)
from .decision import (
    BayesianDecision,
    CandidateSequence,
    CounterfactualDecisionEngine,
)
from .evaluation import (
    ActiveGateReport,
    ActivePairResult,
    CounterfactualGateReport,
    CounterfactualPanel,
    SageTCounterfactualEvaluator,
    active_progress_gate,
    count_forbidden_program_fields,
    counterfactual_gate,
    panels_from_binding_pairs,
)
from .executor import ProgramExecutor, evaluate_expression
from .posterior import ProgramParticle, ProgramPosterior
from .synthesis import (
    AssembledProgram,
    DeterministicFragmentProposer,
    ProgramAssembler,
    ProgramMutator,
)
from .causal.controller import CausalSageTArbitration, CausalSageTController
from .causal.contracts import CausalProgram, CausalState, TransitionEvidence
from .causal.executor import CausalExecutor
from .causal.posterior import CausalPosterior
from .causal.runtime import CausalRuntime

__all__ = [
    "AbstractEntity",
    "AbstractState",
    "ActionBinding",
    "ActionCandidate",
    "ActiveGateReport",
    "ActivePairResult",
    "AssembledProgram",
    "BayesianDecision",
    "CandidateSequence",
    "ConsolidationEntry",
    "ConsolidationRegistry",
    "CounterfactualDecisionEngine",
    "CausalRuntime",
    "CausalExecutor",
    "CausalPosterior",
    "CausalProgram",
    "CausalState",
    "CausalSageTArbitration",
    "CausalSageTController",
    "TransitionEvidence",
    "CounterfactualGateReport",
    "CounterfactualPanel",
    "DeterministicFragmentProposer",
    "Effect",
    "Expression",
    "GoalRule",
    "GroundFact",
    "JointProgramHypothesis",
    "LegacyArbiter",
    "ObjectSchema",
    "ObservedTransition",
    "PredictionPacket",
    "ProgramAssembler",
    "ProgramExecutor",
    "ProgramFragment",
    "ProgramMutator",
    "ProgramParticle",
    "ProgramPosterior",
    "ProgressRule",
    "SageTArbitration",
    "SageTConfig",
    "SageTController",
    "SageTCounterfactualEvaluator",
    "SageTMode",
    "TerminalRule",
    "TransitionRule",
    "TruthValue",
    "active_progress_gate",
    "compile_observation",
    "compile_transition_record",
    "count_forbidden_program_fields",
    "counterfactual_gate",
    "evaluate_expression",
    "panels_from_binding_pairs",
    "program_from_dict",
    "program_to_dict",
]
