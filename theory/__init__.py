"""theory/ — single target package for the 'theory of game' loop.

Step C (this commit): the EPISTEMIC success metric. The belief-revision loop
(step A: MechanicHypothesis / GameTheory / DiscriminatingExperimentDesigner)
will be judged by `score_beliefs` against a `MechanicsOracle`, NOT by game
score. This guards against the previous trap of judging a seductive module
with a bad (ludic) proxy.
"""

from .epistemic_metrics import (
    EpistemicScore,
    GroundTruthFact,
    HypothesisRecord,
    HypothesisStatus,
    MechanicsOracle,
    mechanic_key,
    normalize_operator_kind,
    score_beliefs,
)
from .mechanic_hypothesis import (
    GameTheory,
    MechanicHypothesis,
    ObservedEffect,
    predicted_signal_holds,
)
from .role_hypotheses import (
    ActionRoleHypothesis,
    GoalFamilyHypothesis,
    action_role_key,
    goal_family_key,
    load_task_program_semantic_hypotheses,
    normalize_action_role,
    normalize_goal_family,
)
from .live_transition_loop import (
    LiveTransitionBeliefLoop,
    LiveTransitionUpdate,
    build_observation,
    build_transition_record,
    run_trace_file,
)
from .correspondence_hypothesis import (
    CorrespondenceHypothesis,
    CorrespondenceObservation,
    CorrespondenceRule,
    correspondence_key,
    load_task_program_correspondence_hypotheses,
    predicate_names_for_relation,
)
from .theory_conditioned_planner import (
    PlannerTraceEvent,
    TheoryConditionedPlanner,
    TheoryConditionedPlanningRun,
    TheoryPlan,
    TheoryPlannedAction,
    run_planner_trace_file,
)
from .precondition_hypothesis import (
    PreconditionHypothesis,
    PreconditionObservation,
    extract_precondition_predicates,
    precondition_key,
)
from .theory_option import (
    TheoryOption,
    TheoryOptionInvocation,
    TheoryOptionRun,
    build_options_from_theory,
    run_option_trace_file,
)
from .real_env_option_adapter import (
    EnvFrameSnapshot,
    TheoryOptionRunner,
    TheoryOptionRunnerResult,
    TheoryOptionRunnerRun,
    snapshot_frame,
)
from .correspondence_preparation_policy import (
    PreparationActionHypothesis,
    PreparationPlan,
    PreparationStepResult,
    PrepareCorrespondencePolicy,
    default_ar25_preparation_hypotheses,
)
from .contextual_readiness import (
    ContextualReadinessDiscriminator,
    ContextualReadinessHypothesis,
    ReadinessAssessment,
    ReadinessContext,
    readiness_context,
)
from .strong_preparation_policy import (
    StrongPreparationHypothesis,
    StrongPreparationPlan,
    StrongPreparationStepResult,
    StrongPrepareCorrespondencePolicy,
    default_ar25_strong_preparation_hypotheses,
    discriminating_readiness_gaps,
)
from .prepare_correspondence_option import (
    PrepareCorrespondenceOption,
    PrepareCorrespondenceOptionInvocation,
    PrepareCorrespondenceOptionRunner,
    PrepareCorrespondenceOptionRunnerResult,
)
from .cross_game_correspondence_discovery import (
    CrossGameCorrespondenceDiscoveryRun,
    DiscoveredCorrespondenceCandidate,
    SourceTargetPredicate,
    discover_cross_game_correspondences,
)
from .generic_discriminating_experiment_designer import (
    DesignedExperimentAction,
    DiscriminatingPrediction,
    GenericDiscriminatingExperimentChoice,
    GenericDiscriminatingExperimentDesigner,
)
from .non_ar25_active_micro_run import (
    ActiveExperimentAction,
    NonAr25ActiveMicroRunEvent,
    NonAr25ActiveMicroRunResult,
    run_non_ar25_active_micro_run,
)
from .non_ar25_discriminating_experiment import (
    CompetingHypothesisRevision,
    DiscriminatingNonAr25Experiment,
    NonAr25DiscriminatingExperimentResult,
    run_non_ar25_discriminating_experiment,
)
from .non_ar25_multi_family_experiment import (
    MultiFamilyHypothesisRevision,
    NonAr25MultiFamilyExperimentResult,
    run_non_ar25_multi_family_experiment,
)
from .relation_transfer import (
    RelationTransferPrior,
    RelationTransferRun,
    apply_relation_transfer_priors,
    extract_relation_transfer_priors,
    relation_predictions_from_candidates,
    run_relation_transfer,
)
from .active_transfer_validation import (
    ActiveTransferValidationResult,
    run_active_transfer_validation,
)
from .negative_transfer_memory import (
    NegativeTransferMemory,
    NegativeTransferRecord,
    apply_negative_transfer_memory,
    build_negative_transfer_records,
    transfer_context_signature_from_prediction,
)
from .closed_loop_negative_transfer import (
    ClosedLoopNegativeTransferResult,
    run_closed_loop_negative_transfer,
)
from .relation_option import (
    AvoidRelationOption,
    PrepareRelationOption,
    RelationOption,
    RelationOptionRunResult,
    run_relation_option_micro_run,
)
from .option_composition import (
    OptionCompositionResult,
    run_option_composition,
)
from .multi_relation_option_composition import (
    MultiRelationOptionCompositionResult,
    RelationPreconditionOption,
    default_ar25_relation_preconditions,
    missing_relation_preconditions,
    run_multi_relation_option_composition,
)
from .non_ar25_multi_relation_agenda import (
    NonAr25MultiRelationAgendaResult,
    NonAr25RelationAgendaItem,
    run_non_ar25_multi_relation_agenda,
)
from .non_ar25_functional_progress import (
    FunctionalProgressObservation,
    NonAr25FunctionalProgressResult,
    observe_functional_progress,
    run_non_ar25_functional_progress,
)
from .non_ar25_functional_negative_memory import (
    FunctionalAgendaNegativeMemory,
    FunctionalAgendaNegativeRecord,
    FunctionalNegativeMemoryAgendaResult,
    build_functional_agenda_negative_memory,
    run_functional_negative_memory_agenda,
)
from .non_ar25_transferability_model import (
    NegativeTransferabilityModel,
    TransferabilityGroup,
    TransferabilityModelRunResult,
    build_negative_transferability_model,
    run_negative_transferability_model,
)
from .cross_game_transferability_check import (
    CrossGameTransferabilityCheckResult,
    analogous_target_contexts,
    run_cross_game_transferability_check,
)
from .multi_game_evaluation import (
    EvaluationResult,
    EvaluationTrace,
    MultiGameEvaluationResult,
    evaluate_game,
    run_multi_game_evaluation,
    select_evaluation_traces,
)
from .multi_game_stress_test import (
    MultiGameStressTestResult,
    StressBudgetSummary,
    StressCurvePoint,
    classify_failure,
    run_multi_game_stress_test,
    run_stress_curve_point,
    select_stress_traces,
)
from .unified_cognitive_controller import (
    CognitiveDecision,
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)
from .promoted_relational_rule import (
    PromotedRelationalRule,
    promoted_relational_rule_key,
)
from .online_relational_option import (
    CompiledRelationalOption,
    FunctionalOptionProgress,
    OnlineRelationalOptionCompiler,
    OptionAssessment,
    OptionExecutionMemory,
    OptionOutcomeStats,
    observe_option_progress,
    observed_rule_outcome,
    relation_holds,
)

__all__ = [
    "EpistemicScore",
    "GroundTruthFact",
    "HypothesisRecord",
    "HypothesisStatus",
    "MechanicsOracle",
    "mechanic_key",
    "normalize_operator_kind",
    "score_beliefs",
    "GameTheory",
    "MechanicHypothesis",
    "ObservedEffect",
    "predicted_signal_holds",
    "ActionRoleHypothesis",
    "GoalFamilyHypothesis",
    "action_role_key",
    "goal_family_key",
    "load_task_program_semantic_hypotheses",
    "normalize_action_role",
    "normalize_goal_family",
    "LiveTransitionBeliefLoop",
    "LiveTransitionUpdate",
    "build_observation",
    "build_transition_record",
    "run_trace_file",
    "CorrespondenceHypothesis",
    "CorrespondenceObservation",
    "CorrespondenceRule",
    "correspondence_key",
    "load_task_program_correspondence_hypotheses",
    "predicate_names_for_relation",
    "PlannerTraceEvent",
    "TheoryConditionedPlanner",
    "TheoryConditionedPlanningRun",
    "TheoryPlan",
    "TheoryPlannedAction",
    "run_planner_trace_file",
    "PreconditionHypothesis",
    "PreconditionObservation",
    "extract_precondition_predicates",
    "precondition_key",
    "TheoryOption",
    "TheoryOptionInvocation",
    "TheoryOptionRun",
    "build_options_from_theory",
    "run_option_trace_file",
    "EnvFrameSnapshot",
    "TheoryOptionRunner",
    "TheoryOptionRunnerResult",
    "TheoryOptionRunnerRun",
    "snapshot_frame",
    "PreparationActionHypothesis",
    "PreparationPlan",
    "PreparationStepResult",
    "PrepareCorrespondencePolicy",
    "default_ar25_preparation_hypotheses",
    "ContextualReadinessDiscriminator",
    "ContextualReadinessHypothesis",
    "ReadinessAssessment",
    "ReadinessContext",
    "readiness_context",
    "StrongPreparationHypothesis",
    "StrongPreparationPlan",
    "StrongPreparationStepResult",
    "StrongPrepareCorrespondencePolicy",
    "default_ar25_strong_preparation_hypotheses",
    "discriminating_readiness_gaps",
    "PrepareCorrespondenceOption",
    "PrepareCorrespondenceOptionInvocation",
    "PrepareCorrespondenceOptionRunner",
    "PrepareCorrespondenceOptionRunnerResult",
    "CrossGameCorrespondenceDiscoveryRun",
    "DiscoveredCorrespondenceCandidate",
    "SourceTargetPredicate",
    "discover_cross_game_correspondences",
    "DesignedExperimentAction",
    "DiscriminatingPrediction",
    "GenericDiscriminatingExperimentChoice",
    "GenericDiscriminatingExperimentDesigner",
    "ActiveExperimentAction",
    "NonAr25ActiveMicroRunEvent",
    "NonAr25ActiveMicroRunResult",
    "run_non_ar25_active_micro_run",
    "CompetingHypothesisRevision",
    "DiscriminatingNonAr25Experiment",
    "NonAr25DiscriminatingExperimentResult",
    "run_non_ar25_discriminating_experiment",
    "MultiFamilyHypothesisRevision",
    "NonAr25MultiFamilyExperimentResult",
    "run_non_ar25_multi_family_experiment",
    "RelationTransferPrior",
    "RelationTransferRun",
    "apply_relation_transfer_priors",
    "extract_relation_transfer_priors",
    "relation_predictions_from_candidates",
    "run_relation_transfer",
    "ActiveTransferValidationResult",
    "run_active_transfer_validation",
    "NegativeTransferMemory",
    "NegativeTransferRecord",
    "apply_negative_transfer_memory",
    "build_negative_transfer_records",
    "transfer_context_signature_from_prediction",
    "ClosedLoopNegativeTransferResult",
    "run_closed_loop_negative_transfer",
    "AvoidRelationOption",
    "PrepareRelationOption",
    "RelationOption",
    "RelationOptionRunResult",
    "run_relation_option_micro_run",
    "OptionCompositionResult",
    "run_option_composition",
    "MultiRelationOptionCompositionResult",
    "RelationPreconditionOption",
    "default_ar25_relation_preconditions",
    "missing_relation_preconditions",
    "run_multi_relation_option_composition",
    "NonAr25MultiRelationAgendaResult",
    "NonAr25RelationAgendaItem",
    "run_non_ar25_multi_relation_agenda",
    "FunctionalProgressObservation",
    "NonAr25FunctionalProgressResult",
    "observe_functional_progress",
    "run_non_ar25_functional_progress",
    "FunctionalAgendaNegativeMemory",
    "FunctionalAgendaNegativeRecord",
    "FunctionalNegativeMemoryAgendaResult",
    "build_functional_agenda_negative_memory",
    "run_functional_negative_memory_agenda",
    "NegativeTransferabilityModel",
    "TransferabilityGroup",
    "TransferabilityModelRunResult",
    "build_negative_transferability_model",
    "run_negative_transferability_model",
    "CrossGameTransferabilityCheckResult",
    "analogous_target_contexts",
    "run_cross_game_transferability_check",
    "EvaluationResult",
    "EvaluationTrace",
    "MultiGameEvaluationResult",
    "evaluate_game",
    "run_multi_game_evaluation",
    "select_evaluation_traces",
    "MultiGameStressTestResult",
    "StressBudgetSummary",
    "StressCurvePoint",
    "classify_failure",
    "run_multi_game_stress_test",
    "run_stress_curve_point",
    "select_stress_traces",
    "CognitiveDecision",
    "UnifiedCognitiveConfig",
    "UnifiedCognitiveController",
    "PromotedRelationalRule",
    "promoted_relational_rule_key",
    "CompiledRelationalOption",
    "FunctionalOptionProgress",
    "OnlineRelationalOptionCompiler",
    "OptionAssessment",
    "OptionExecutionMemory",
    "OptionOutcomeStats",
    "observe_option_progress",
    "observed_rule_outcome",
    "relation_holds",
]
