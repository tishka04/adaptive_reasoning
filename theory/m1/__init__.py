"""M1 symbolic imagination package.

M1 produces unresolved hypothesis material from raw observations. It must not
confirm anything; confirmation remains the job of the A15-A31 scientific loop.
"""

__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_TRACES_DIR",
    "InvariantMiningResult",
    "LatentInvariant",
    "PredicateCoveragePretest",
    "PredicateInjectionRule",
    "ExpandedAnchor",
    "LiveAnchorCompatibilityPretest",
    "LiveAnchorCompatibilityMetrics",
    "DEFAULT_GROUNDING_AUTOPSY_JSON_PATH",
    "DEFAULT_GROUNDING_AUTOPSY_MD_PATH",
    "DEFAULT_ACTIONABLE_ALIGNMENT_OUTPUT_PATH",
    "DEFAULT_MECHANIC_TYPING_OUTPUT_PATH",
    "DEFAULT_MECHANIC_GROUNDED_CANDIDATES_OUTPUT_PATH",
    "DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH",
    "DEFAULT_POLYMORPHIC_A25_ADAPTER_OUTPUT_PATH",
    "DEFAULT_EXPERIMENT_VALUE_ESTIMATES_OUTPUT_PATH",
    "DEFAULT_RECOMMENDED_EXPERIMENT_CHOICE_OUTPUT_PATH",
    "DEFAULT_MECHANIC_REVISION_CANDIDATES_OUTPUT_PATH",
    "DEFAULT_SCIENTIFIC_INTEGRATION_PRETEST_OUTPUT_PATH",
    "DEFAULT_CONTROLLED_EXPERIMENT_RESULTS_OUTPUT_PATH",
    "DEFAULT_SOURCE_REACHABILITY_OUTPUT_PATH",
    "ConcretePolymorphicAction",
    "ControlledExperiment",
    "ExperimentalPrecondition",
    "ExperimentalValueEstimate",
    "GameMechanicProfile",
    "GroundingFunnel",
    "MechanicGroundedExperimentCandidate",
    "MechanicHypothesisCandidate",
    "MechanicObservation",
    "MechanicPredictionCandidate",
    "MechanicRevisionCandidate",
    "PreparationAction",
    "PolymorphicA25ExperimentalChoice",
    "PolymorphicMechanicExperiment",
    "PolymorphicA25PretestRow",
    "SearchResult",
    "ScientificLedgerEntry",
    "SourceAlignmentProblem",
    "source_alignment_problem_from_dict",
    "run_a31bis",
    "RawTransitionObservation",
    "actionable_invariant_rules",
    "build_m1_anchor_expander",
    "build_m1_live_candidate_ranker",
    "build_m1_predicate_generator",
    "build_dataset",
    "execute_polymorphic_candidate",
    "estimate_experiment_values",
    "find_experimental_precondition",
    "generate_predicates",
    "expand_anchors",
    "grounding_block_reason",
    "historical_predicates",
    "extract_source_alignment_problems",
    "load_invariants_json",
    "load_grounding_autopsy",
    "iter_trace_observations",
    "load_observations_jsonl",
    "load_actionable_invariant_rules",
    "measure_required_observation",
    "mine_invariants",
    "raw_attribute_outcomes",
    "run_predicate_coverage_pretest",
    "run_live_anchor_compatibility_pretest",
    "run_grounding_autopsy",
    "run_actionable_source_alignment_pretest",
    "run_source_reachability_analysis",
    "run_mechanic_typing",
    "run_mechanic_grounded_candidate_generation",
    "run_polymorphic_a25_pretest",
    "run_minimal_polymorphic_a25_adapter",
    "run_experiment_value_estimation",
    "run_recommended_polymorphic_a25_choice",
    "run_mechanic_revision_candidate_generation",
    "run_scientific_integration_pretest",
    "run_controlled_followup_experiment",
    "select_blocked_trace_paths",
    "select_testable_candidates",
    "summarize_polymorphic_pretest",
    "summarize_polymorphic_adapter_results",
    "summarize_experiment_value_estimates",
    "summarize_recommended_choice",
    "summarize_revision_candidates",
    "summarize_scientific_integration",
    "summarize_controlled_experiments",
    "summarize_source_alignment_problems",
    "summarize_profiles",
    "summarize_candidates",
    "summarize_stress_dict",
    "summarize_raw_outcome_rates",
    "write_a31bis_result",
    "write_actionable_source_alignment_pretest",
    "write_grounding_autopsy",
    "write_mechanic_typing",
    "write_mechanic_grounded_candidates",
    "write_polymorphic_a25_pretest",
    "write_polymorphic_a25_adapter_result",
    "write_experiment_value_estimates",
    "write_recommended_polymorphic_a25_choice",
    "write_mechanic_revision_candidates",
    "write_scientific_integration_pretest",
    "write_controlled_experiment_results",
    "write_live_anchor_compatibility_pretest",
    "write_source_reachability_analysis",
    "write_invariant_outputs",
    "write_observations_jsonl",
    "write_pretest_result",
]


def __getattr__(name: str):
    observation_names = {
        "DEFAULT_OUTPUT_PATH",
        "DEFAULT_TRACES_DIR",
        "RawTransitionObservation",
        "build_dataset",
        "iter_trace_observations",
        "load_observations_jsonl",
        "write_observations_jsonl",
    }
    miner_names = {
        "InvariantMiningResult",
        "LatentInvariant",
        "load_invariants_json",
        "mine_invariants",
        "raw_attribute_outcomes",
        "summarize_raw_outcome_rates",
        "write_invariant_outputs",
    }
    predicate_names = {
        "PredicateCoveragePretest",
        "PredicateInjectionRule",
        "actionable_invariant_rules",
        "build_m1_predicate_generator",
        "generate_predicates",
        "historical_predicates",
        "load_actionable_invariant_rules",
        "run_predicate_coverage_pretest",
        "select_blocked_trace_paths",
        "write_pretest_result",
    }
    anchor_names = {
        "ExpandedAnchor",
        "build_m1_anchor_expander",
        "expand_anchors",
    }
    live_anchor_names = {
        "LiveAnchorCompatibilityMetrics",
        "LiveAnchorCompatibilityPretest",
        "build_m1_live_candidate_ranker",
        "run_live_anchor_compatibility_pretest",
        "write_live_anchor_compatibility_pretest",
    }
    grounding_autopsy_names = {
        "DEFAULT_GROUNDING_AUTOPSY_JSON_PATH",
        "DEFAULT_GROUNDING_AUTOPSY_MD_PATH",
        "GroundingFunnel",
        "grounding_block_reason",
        "run_grounding_autopsy",
        "write_grounding_autopsy",
    }
    source_reachability_names = {
        "DEFAULT_SOURCE_REACHABILITY_OUTPUT_PATH",
        "SourceAlignmentProblem",
        "extract_source_alignment_problems",
        "load_grounding_autopsy",
        "run_source_reachability_analysis",
        "source_alignment_problem_from_dict",
        "summarize_source_alignment_problems",
        "write_source_reachability_analysis",
    }
    actionable_source_alignment_names = {
        "DEFAULT_ACTIONABLE_ALIGNMENT_OUTPUT_PATH",
        "ExperimentalPrecondition",
        "PreparationAction",
        "SearchResult",
        "find_experimental_precondition",
        "run_actionable_source_alignment_pretest",
        "write_actionable_source_alignment_pretest",
    }
    mechanic_typing_names = {
        "DEFAULT_MECHANIC_TYPING_OUTPUT_PATH",
        "GameMechanicProfile",
        "run_mechanic_typing",
        "summarize_profiles",
        "write_mechanic_typing",
    }
    mechanic_grounded_candidate_names = {
        "DEFAULT_MECHANIC_GROUNDED_CANDIDATES_OUTPUT_PATH",
        "MechanicGroundedExperimentCandidate",
        "run_mechanic_grounded_candidate_generation",
        "summarize_candidates",
        "write_mechanic_grounded_candidates",
    }
    polymorphic_a25_pretest_names = {
        "DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH",
        "PolymorphicA25PretestRow",
        "run_polymorphic_a25_pretest",
        "summarize_polymorphic_pretest",
        "write_polymorphic_a25_pretest",
    }
    polymorphic_a25_adapter_names = {
        "DEFAULT_POLYMORPHIC_A25_ADAPTER_OUTPUT_PATH",
        "ConcretePolymorphicAction",
        "PolymorphicMechanicExperiment",
        "execute_polymorphic_candidate",
        "measure_required_observation",
        "run_minimal_polymorphic_a25_adapter",
        "select_testable_candidates",
        "summarize_polymorphic_adapter_results",
        "write_polymorphic_a25_adapter_result",
    }
    experiment_value_estimator_names = {
        "DEFAULT_EXPERIMENT_VALUE_ESTIMATES_OUTPUT_PATH",
        "ExperimentalValueEstimate",
        "estimate_experiment_values",
        "run_experiment_value_estimation",
        "summarize_experiment_value_estimates",
        "write_experiment_value_estimates",
    }
    recommended_experiment_choice_names = {
        "DEFAULT_RECOMMENDED_EXPERIMENT_CHOICE_OUTPUT_PATH",
        "MechanicHypothesisCandidate",
        "MechanicObservation",
        "PolymorphicA25ExperimentalChoice",
        "run_recommended_polymorphic_a25_choice",
        "summarize_recommended_choice",
        "write_recommended_polymorphic_a25_choice",
    }
    mechanic_revision_candidate_names = {
        "DEFAULT_MECHANIC_REVISION_CANDIDATES_OUTPUT_PATH",
        "MechanicPredictionCandidate",
        "MechanicRevisionCandidate",
        "run_mechanic_revision_candidate_generation",
        "summarize_revision_candidates",
        "write_mechanic_revision_candidates",
    }
    scientific_integration_pretest_names = {
        "DEFAULT_SCIENTIFIC_INTEGRATION_PRETEST_OUTPUT_PATH",
        "ScientificLedgerEntry",
        "run_scientific_integration_pretest",
        "summarize_scientific_integration",
        "write_scientific_integration_pretest",
    }
    controlled_followup_experiment_names = {
        "DEFAULT_CONTROLLED_EXPERIMENT_RESULTS_OUTPUT_PATH",
        "ControlledExperiment",
        "run_controlled_followup_experiment",
        "summarize_controlled_experiments",
        "write_controlled_experiment_results",
    }
    stress_names = {
        "run_a31bis",
        "summarize_stress_dict",
        "write_a31bis_result",
    }
    if name in observation_names:
        import importlib

        observation_dataset = importlib.import_module(".observation_dataset", __name__)
        return getattr(observation_dataset, name)
    if name in miner_names:
        import importlib

        invariant_miner = importlib.import_module(".invariant_miner", __name__)
        return getattr(invariant_miner, name)
    if name in predicate_names:
        import importlib

        predicate_generation = importlib.import_module(".predicate_generation", __name__)
        return getattr(predicate_generation, name)
    if name in anchor_names:
        import importlib

        anchor_expansion = importlib.import_module(".anchor_expansion", __name__)
        return getattr(anchor_expansion, name)
    if name in live_anchor_names:
        import importlib

        live_anchor_ranking = importlib.import_module(".live_anchor_ranking", __name__)
        return getattr(live_anchor_ranking, name)
    if name in grounding_autopsy_names:
        import importlib

        grounding_autopsy = importlib.import_module(".grounding_autopsy", __name__)
        return getattr(grounding_autopsy, name)
    if name in source_reachability_names:
        import importlib

        source_reachability = importlib.import_module(".source_reachability", __name__)
        return getattr(source_reachability, name)
    if name in actionable_source_alignment_names:
        import importlib

        actionable_source_alignment = importlib.import_module(
            ".actionable_source_alignment",
            __name__,
        )
        return getattr(actionable_source_alignment, name)
    if name in mechanic_typing_names:
        import importlib

        mechanic_typing = importlib.import_module(".mechanic_typing", __name__)
        return getattr(mechanic_typing, name)
    if name in mechanic_grounded_candidate_names:
        import importlib

        mechanic_grounded_candidates = importlib.import_module(
            ".mechanic_grounded_candidates",
            __name__,
        )
        return getattr(mechanic_grounded_candidates, name)
    if name in polymorphic_a25_pretest_names:
        import importlib

        polymorphic_a25_pretest = importlib.import_module(
            ".polymorphic_a25_pretest",
            __name__,
        )
        return getattr(polymorphic_a25_pretest, name)
    if name in polymorphic_a25_adapter_names:
        import importlib

        polymorphic_a25_adapter = importlib.import_module(
            ".polymorphic_a25_adapter",
            __name__,
        )
        return getattr(polymorphic_a25_adapter, name)
    if name in experiment_value_estimator_names:
        import importlib

        experiment_value_estimator = importlib.import_module(
            ".experiment_value_estimator",
            __name__,
        )
        return getattr(experiment_value_estimator, name)
    if name in recommended_experiment_choice_names:
        import importlib

        recommended_experiment_choice = importlib.import_module(
            ".recommended_experiment_choice",
            __name__,
        )
        return getattr(recommended_experiment_choice, name)
    if name in mechanic_revision_candidate_names:
        import importlib

        mechanic_revision_candidate = importlib.import_module(
            ".mechanic_revision_candidate",
            __name__,
        )
        return getattr(mechanic_revision_candidate, name)
    if name in scientific_integration_pretest_names:
        import importlib

        scientific_integration_pretest = importlib.import_module(
            ".scientific_integration_pretest",
            __name__,
        )
        return getattr(scientific_integration_pretest, name)
    if name in controlled_followup_experiment_names:
        import importlib

        controlled_followup_experiment = importlib.import_module(
            ".controlled_followup_experiment",
            __name__,
        )
        return getattr(controlled_followup_experiment, name)
    if name in stress_names:
        import importlib

        stress_test_a31bis = importlib.import_module(".stress_test_a31bis", __name__)
        return getattr(stress_test_a31bis, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
