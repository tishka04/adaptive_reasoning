"""M3 minimal scientific planning package."""

__all__ = [
    "CONTROL_REUSE_REASON",
    "DEFAULT_M3_BUDGET",
    "DEFAULT_M3_GAME_ID",
    "DEFAULT_M3_OUTPUT_PATH",
    "DEFAULT_PREFERRED_CONTROLS",
    "DEFAULT_A15_REVISION_QUEUE_OUTPUT_PATH",
    "DEFAULT_REFINED_FOLLOWUP_REQUESTS_OUTPUT_PATH",
    "DEFAULT_REFINED_FOLLOWUP_RESULTS_OUTPUT_PATH",
    "DEFAULT_DYNAMIC_RETARGET_REQUESTS_OUTPUT_PATH",
    "DEFAULT_DYNAMIC_RETARGET_RESULTS_OUTPUT_PATH",
    "DEFAULT_DYNAMIC_RETARGET_MECHANISM_CANDIDATES_OUTPUT_PATH",
    "DEFAULT_DYNAMIC_RETARGET_SELECTION_RULES_OUTPUT_PATH",
    "DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_REQUESTS_OUTPUT_PATH",
    "DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_RESULTS_OUTPUT_PATH",
    "DEFAULT_DYNAMIC_RETARGET_SELECTION_RULE_CONSOLIDATION_OUTPUT_PATH",
    "DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_REQUESTS_OUTPUT_PATH",
    "DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_RESULTS_OUTPUT_PATH",
    "DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_GENERATIVITY_CONSOLIDATION_OUTPUT_PATH",
    "DEFAULT_PATCH_SIMILARITY_A32_REVISION_QUEUE_OUTPUT_PATH",
    "DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_REQUESTS_OUTPUT_PATH",
    "DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_RESULTS_OUTPUT_PATH",
    "DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH",
    "A15RevisionQueueItem",
    "DynamicRetargetMechanismCandidate",
    "DynamicRetargetFollowupRequest",
    "PlannedControlledExperiment",
    "RefinedFollowupExperimentRequest",
    "RetargetSelectionRuleSet",
    "RetargetSelectionRuleConsolidation",
    "PatchSimilarityExpansionRequest",
    "PatchSimilarityGenerativityConsolidation",
    "PatchSimilarityA32RevisionQueueItem",
    "A32RequestedPatchSimilarityFollowupRequest",
    "SelectionRuleFollowupRequest",
    "ScientificPlanningState",
    "available_controls_for_target",
    "build_a15_revision_queue_items",
    "build_followup_request_for_refined_hypothesis",
    "build_scientific_planning_state",
    "build_scientific_planning_state_from_payloads",
    "candidate_record_dict",
    "choose_control_action",
    "execute_planned_controlled_experiment",
    "generate_dynamic_retarget_candidates",
    "load_live_available_action_names",
    "run_a15_revision_queue_generation",
    "run_dynamic_retarget_followup_planning",
    "run_dynamic_retarget_followup_execution",
    "run_dynamic_retarget_mechanism_consolidation",
    "run_dynamic_retarget_selection_rule_induction",
    "run_dynamic_retarget_selection_followup_planning",
    "run_dynamic_retarget_selection_followup_execution",
    "run_dynamic_retarget_selection_rule_consolidation",
    "run_dynamic_retarget_patch_similarity_expansion_planning",
    "run_dynamic_retarget_patch_similarity_expansion_execution",
    "run_dynamic_retarget_patch_similarity_generativity_consolidation",
    "run_patch_similarity_a32_revision_queue_generation",
    "run_a32_requested_patch_similarity_followup_planning",
    "run_a32_requested_patch_similarity_followup_execution",
    "run_a32_requested_patch_similarity_scope_consolidation",
    "run_refined_followup_execution",
    "run_refined_followup_planning",
    "run_scientific_planning_loop",
    "resolve_target_action_args",
    "select_next_experiment",
    "summarize_a15_revision_queue",
    "summarize_followup_requests",
    "summarize_scientific_planning_loop",
    "updated_ledger_entries_from_state",
    "write_a15_revision_queue",
    "write_dynamic_retarget_followup_requests",
    "write_dynamic_retarget_followup_results",
    "write_dynamic_retarget_mechanism_candidates",
    "write_dynamic_retarget_selection_rules",
    "write_dynamic_retarget_selection_followup_requests",
    "write_dynamic_retarget_selection_followup_results",
    "write_dynamic_retarget_selection_rule_consolidation",
    "write_dynamic_retarget_patch_similarity_expansion_requests",
    "write_dynamic_retarget_patch_similarity_expansion_results",
    "write_dynamic_retarget_patch_similarity_generativity_consolidation",
    "write_patch_similarity_a32_revision_queue",
    "write_a32_requested_patch_similarity_followup_requests",
    "write_a32_requested_patch_similarity_followup_results",
    "write_a32_requested_patch_similarity_scope_consolidation",
    "write_refined_followup_requests",
    "write_refined_followup_results",
    "write_scientific_planning_result",
]


def __getattr__(name: str):
    state_names = {
        "ScientificPlanningState",
        "build_scientific_planning_state",
        "build_scientific_planning_state_from_payloads",
        "updated_ledger_entries_from_state",
    }
    selector_names = {
        "CONTROL_REUSE_REASON",
        "DEFAULT_PREFERRED_CONTROLS",
        "PlannedControlledExperiment",
        "available_controls_for_target",
        "choose_control_action",
        "select_next_experiment",
    }
    loop_names = {
        "DEFAULT_M3_BUDGET",
        "DEFAULT_M3_GAME_ID",
        "DEFAULT_M3_OUTPUT_PATH",
        "execute_planned_controlled_experiment",
        "load_live_available_action_names",
        "run_scientific_planning_loop",
        "summarize_scientific_planning_loop",
        "write_scientific_planning_result",
    }
    queue_names = {
        "DEFAULT_A15_REVISION_QUEUE_OUTPUT_PATH",
        "A15RevisionQueueItem",
        "build_a15_revision_queue_items",
        "candidate_record_dict",
        "run_a15_revision_queue_generation",
        "summarize_a15_revision_queue",
        "write_a15_revision_queue",
    }
    followup_names = {
        "DEFAULT_REFINED_FOLLOWUP_REQUESTS_OUTPUT_PATH",
        "RefinedFollowupExperimentRequest",
        "build_followup_request_for_refined_hypothesis",
        "run_refined_followup_planning",
        "summarize_followup_requests",
        "write_refined_followup_requests",
    }
    followup_execution_names = {
        "DEFAULT_REFINED_FOLLOWUP_RESULTS_OUTPUT_PATH",
        "resolve_target_action_args",
        "run_refined_followup_execution",
        "write_refined_followup_results",
    }
    dynamic_retarget_names = {
        "DEFAULT_DYNAMIC_RETARGET_REQUESTS_OUTPUT_PATH",
        "DynamicRetargetFollowupRequest",
        "generate_dynamic_retarget_candidates",
        "run_dynamic_retarget_followup_planning",
        "write_dynamic_retarget_followup_requests",
    }
    dynamic_retarget_execution_names = {
        "DEFAULT_DYNAMIC_RETARGET_RESULTS_OUTPUT_PATH",
        "run_dynamic_retarget_followup_execution",
        "write_dynamic_retarget_followup_results",
    }
    dynamic_retarget_mechanism_names = {
        "DEFAULT_DYNAMIC_RETARGET_MECHANISM_CANDIDATES_OUTPUT_PATH",
        "DynamicRetargetMechanismCandidate",
        "run_dynamic_retarget_mechanism_consolidation",
        "write_dynamic_retarget_mechanism_candidates",
    }
    dynamic_retarget_selection_rule_names = {
        "DEFAULT_DYNAMIC_RETARGET_SELECTION_RULES_OUTPUT_PATH",
        "RetargetSelectionRuleSet",
        "run_dynamic_retarget_selection_rule_induction",
        "write_dynamic_retarget_selection_rules",
    }
    dynamic_retarget_selection_followup_names = {
        "DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_REQUESTS_OUTPUT_PATH",
        "SelectionRuleFollowupRequest",
        "run_dynamic_retarget_selection_followup_planning",
        "write_dynamic_retarget_selection_followup_requests",
    }
    dynamic_retarget_selection_execution_names = {
        "DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_RESULTS_OUTPUT_PATH",
        "run_dynamic_retarget_selection_followup_execution",
        "write_dynamic_retarget_selection_followup_results",
    }
    dynamic_retarget_selection_consolidation_names = {
        "DEFAULT_DYNAMIC_RETARGET_SELECTION_RULE_CONSOLIDATION_OUTPUT_PATH",
        "RetargetSelectionRuleConsolidation",
        "run_dynamic_retarget_selection_rule_consolidation",
        "write_dynamic_retarget_selection_rule_consolidation",
    }
    dynamic_retarget_patch_expansion_names = {
        "DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_REQUESTS_OUTPUT_PATH",
        "PatchSimilarityExpansionRequest",
        "run_dynamic_retarget_patch_similarity_expansion_planning",
        "write_dynamic_retarget_patch_similarity_expansion_requests",
    }
    dynamic_retarget_patch_expansion_execution_names = {
        "DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_RESULTS_OUTPUT_PATH",
        "run_dynamic_retarget_patch_similarity_expansion_execution",
        "write_dynamic_retarget_patch_similarity_expansion_results",
    }
    dynamic_retarget_patch_generativity_names = {
        "DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_GENERATIVITY_CONSOLIDATION_OUTPUT_PATH",
        "PatchSimilarityGenerativityConsolidation",
        "run_dynamic_retarget_patch_similarity_generativity_consolidation",
        "write_dynamic_retarget_patch_similarity_generativity_consolidation",
    }
    patch_similarity_a32_queue_names = {
        "DEFAULT_PATCH_SIMILARITY_A32_REVISION_QUEUE_OUTPUT_PATH",
        "PatchSimilarityA32RevisionQueueItem",
        "run_patch_similarity_a32_revision_queue_generation",
        "write_patch_similarity_a32_revision_queue",
    }
    a32_requested_patch_followup_names = {
        "DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_REQUESTS_OUTPUT_PATH",
        "A32RequestedPatchSimilarityFollowupRequest",
        "run_a32_requested_patch_similarity_followup_planning",
        "write_a32_requested_patch_similarity_followup_requests",
    }
    a32_requested_patch_followup_execution_names = {
        "DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_RESULTS_OUTPUT_PATH",
        "run_a32_requested_patch_similarity_followup_execution",
        "write_a32_requested_patch_similarity_followup_results",
    }
    a32_requested_patch_scope_consolidation_names = {
        "DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH",
        "run_a32_requested_patch_similarity_scope_consolidation",
        "write_a32_requested_patch_similarity_scope_consolidation",
    }
    if name in state_names:
        import importlib

        module = importlib.import_module(".scientific_planner_state", __name__)
        return getattr(module, name)
    if name in selector_names:
        import importlib

        module = importlib.import_module(".next_experiment_selector", __name__)
        return getattr(module, name)
    if name in loop_names:
        import importlib

        module = importlib.import_module(".scientific_planning_loop", __name__)
        return getattr(module, name)
    if name in queue_names:
        import importlib

        module = importlib.import_module(".a15_revision_queue", __name__)
        return getattr(module, name)
    if name in followup_names:
        import importlib

        module = importlib.import_module(".refined_followup_planner", __name__)
        return getattr(module, name)
    if name in followup_execution_names:
        import importlib

        module = importlib.import_module(".refined_followup_executor", __name__)
        return getattr(module, name)
    if name in dynamic_retarget_names:
        import importlib

        module = importlib.import_module(".dynamic_retarget_followup_planner", __name__)
        return getattr(module, name)
    if name in dynamic_retarget_execution_names:
        import importlib

        module = importlib.import_module(".dynamic_retarget_followup_executor", __name__)
        return getattr(module, name)
    if name in dynamic_retarget_mechanism_names:
        import importlib

        module = importlib.import_module(
            ".dynamic_retarget_mechanism_consolidation", __name__
        )
        return getattr(module, name)
    if name in dynamic_retarget_selection_rule_names:
        import importlib

        module = importlib.import_module(
            ".dynamic_retarget_selection_rule_induction", __name__
        )
        return getattr(module, name)
    if name in dynamic_retarget_selection_followup_names:
        import importlib

        module = importlib.import_module(
            ".dynamic_retarget_selection_followup_planner", __name__
        )
        return getattr(module, name)
    if name in dynamic_retarget_selection_execution_names:
        import importlib

        module = importlib.import_module(
            ".dynamic_retarget_selection_followup_executor", __name__
        )
        return getattr(module, name)
    if name in dynamic_retarget_selection_consolidation_names:
        import importlib

        module = importlib.import_module(
            ".dynamic_retarget_selection_rule_consolidation", __name__
        )
        return getattr(module, name)
    if name in dynamic_retarget_patch_expansion_names:
        import importlib

        module = importlib.import_module(
            ".dynamic_retarget_patch_similarity_expansion_planner", __name__
        )
        return getattr(module, name)
    if name in dynamic_retarget_patch_expansion_execution_names:
        import importlib

        module = importlib.import_module(
            ".dynamic_retarget_patch_similarity_expansion_executor", __name__
        )
        return getattr(module, name)
    if name in dynamic_retarget_patch_generativity_names:
        import importlib

        module = importlib.import_module(
            ".dynamic_retarget_patch_similarity_generativity_consolidation", __name__
        )
        return getattr(module, name)
    if name in patch_similarity_a32_queue_names:
        import importlib

        module = importlib.import_module(
            ".patch_similarity_a32_revision_queue", __name__
        )
        return getattr(module, name)
    if name in a32_requested_patch_followup_names:
        import importlib

        module = importlib.import_module(
            ".a32_requested_patch_similarity_followup_planner", __name__
        )
        return getattr(module, name)
    if name in a32_requested_patch_followup_execution_names:
        import importlib

        module = importlib.import_module(
            ".a32_requested_patch_similarity_followup_executor", __name__
        )
        return getattr(module, name)
    if name in a32_requested_patch_scope_consolidation_names:
        import importlib

        module = importlib.import_module(
            ".a32_requested_patch_similarity_scope_consolidation", __name__
        )
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
