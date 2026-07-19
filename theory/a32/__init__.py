"""A32 scientific revision decision package."""

__all__ = [
    "A32RevisionDecision",
    "A32PatchSimilarityRevisionIntakeCandidate",
    "A32PatchSimilarityRevisionDecision",
    "A32UnknownGameControlProtocolDecision",
    "A32UnknownGameParameterizedControlRevisionDecision",
    "A32SecondUnknownGameControlDependenceRevisionDecision",
    "A32ThirdUnknownGameParameterizedRelationDecision",
    "DEFAULT_A32_REVISION_DECISIONS_OUTPUT_PATH",
    "DEFAULT_A32_PATCH_SIMILARITY_REVISION_INTAKE_OUTPUT_PATH",
    "DEFAULT_A32_PATCH_SIMILARITY_REVISION_DECISIONS_OUTPUT_PATH",
    "DEFAULT_A32_UNKNOWN_GAME_CONTROL_PROTOCOL_DECISIONS_OUTPUT_PATH",
    "DEFAULT_A32_UNKNOWN_GAME_PARAMETERIZED_CONTROL_REVISION_OUTPUT_PATH",
    "DEFAULT_A32_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_REVISION_OUTPUT_PATH",
    "DEFAULT_A32_THIRD_UNKNOWN_GAME_PARAMETERIZED_RELATION_REVISION_PATH",
    "FOLLOWUP_REQUIRED",
    "REVISION_ACCEPTED_AS_CONFIRMED",
    "REVISION_REJECTED_AS_INSUFFICIENT",
    "ACCEPTED_FOR_SCIENTIFIC_REVISION",
    "CONFIRM_AFTER_SCOPE_LIMITED_REVISION",
    "REFUTE_AFTER_REVISION",
    "REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS",
    "REJECTED_FROM_INTAKE",
    "SCOPE_LIMITED_CANDIDATE_ONLY",
    "AUTHORIZE_PARAMETERIZED_CONTROL_PROTOCOL",
    "RETAIN_STRICT_ACTION_DISTINCT_REQUIREMENT",
    "REJECT_UNIDENTIFIABLE_CURRENT_ACTION_SURFACE",
    "CONFIRM_SCOPE_LIMITED_AFTER_PARAMETERIZED_CONTROL_REVISION",
    "KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_CONTROL",
    "REFUTE_AFTER_PARAMETERIZED_CONTROL_CONTRADICTION",
    "REQUEST_MORE_TESTS_AFTER_INCOMPLETE_PARAMETERIZED_CONTROL",
    "CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST",
    "KEEP_STANDALONE_ACTION2_EFFECT_UNRESOLVED_NON_IDENTIFIABLE",
    "REFUTE_CONTROL_DEPENDENT_CONTRAST",
    "REQUEST_MORE_TESTS_FOR_CONTROL_DEPENDENCE",
    "CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_PARAMETERIZED_RELATION",
    "KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_TARGET_EFFECT",
    "REFUTE_CONTROL_DEPENDENT_PARAMETERIZED_RELATION",
    "REQUEST_MORE_TESTS_FOR_PARAMETERIZED_RELATION",
    "build_a32_patch_similarity_revision_decisions",
    "build_a32_patch_similarity_revision_intake_candidates",
    "build_a32_revision_decisions",
    "build_a32_unknown_game_control_protocol_decisions",
    "build_a32_unknown_game_parameterized_control_revision_decisions",
    "build_a32_second_unknown_game_control_dependence_revision_decisions",
    "build_a32_third_unknown_game_parameterized_relation_decisions",
    "run_a32_patch_similarity_revision_decision_consumer",
    "run_a32_patch_similarity_revision_intake",
    "run_a32_revision_decision_consumer",
    "run_a32_unknown_game_control_protocol_decision_consumer",
    "run_a32_unknown_game_parameterized_control_revision_consumer",
    "run_a32_second_unknown_game_control_dependence_revision_consumer",
    "run_a32_third_unknown_game_parameterized_relation_revision",
    "write_a32_patch_similarity_revision_decisions",
    "write_a32_patch_similarity_revision_intake",
    "write_a32_revision_decisions",
    "write_a32_unknown_game_control_protocol_decisions",
    "write_a32_unknown_game_parameterized_control_revision_decisions",
    "write_a32_second_unknown_game_control_dependence_revision_decisions",
    "write_a32_third_unknown_game_parameterized_relation_revision",
]


def __getattr__(name: str):
    patch_intake_names = {
        "A32PatchSimilarityRevisionIntakeCandidate",
        "DEFAULT_A32_PATCH_SIMILARITY_REVISION_INTAKE_OUTPUT_PATH",
        "ACCEPTED_FOR_SCIENTIFIC_REVISION",
        "REJECTED_FROM_INTAKE",
        "build_a32_patch_similarity_revision_intake_candidates",
        "run_a32_patch_similarity_revision_intake",
        "write_a32_patch_similarity_revision_intake",
    }
    patch_decision_names = {
        "A32PatchSimilarityRevisionDecision",
        "DEFAULT_A32_PATCH_SIMILARITY_REVISION_DECISIONS_OUTPUT_PATH",
        "CONFIRM_AFTER_SCOPE_LIMITED_REVISION",
        "REFUTE_AFTER_REVISION",
        "REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS",
        "SCOPE_LIMITED_CANDIDATE_ONLY",
        "build_a32_patch_similarity_revision_decisions",
        "run_a32_patch_similarity_revision_decision_consumer",
        "write_a32_patch_similarity_revision_decisions",
    }
    unknown_game_protocol_names = {
        "A32UnknownGameControlProtocolDecision",
        "DEFAULT_A32_UNKNOWN_GAME_CONTROL_PROTOCOL_DECISIONS_OUTPUT_PATH",
        "AUTHORIZE_PARAMETERIZED_CONTROL_PROTOCOL",
        "RETAIN_STRICT_ACTION_DISTINCT_REQUIREMENT",
        "REJECT_UNIDENTIFIABLE_CURRENT_ACTION_SURFACE",
        "build_a32_unknown_game_control_protocol_decisions",
        "run_a32_unknown_game_control_protocol_decision_consumer",
        "write_a32_unknown_game_control_protocol_decisions",
    }
    unknown_game_parameterized_revision_names = {
        "A32UnknownGameParameterizedControlRevisionDecision",
        "DEFAULT_A32_UNKNOWN_GAME_PARAMETERIZED_CONTROL_REVISION_OUTPUT_PATH",
        "CONFIRM_SCOPE_LIMITED_AFTER_PARAMETERIZED_CONTROL_REVISION",
        "KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_CONTROL",
        "REFUTE_AFTER_PARAMETERIZED_CONTROL_CONTRADICTION",
        "REQUEST_MORE_TESTS_AFTER_INCOMPLETE_PARAMETERIZED_CONTROL",
        "build_a32_unknown_game_parameterized_control_revision_decisions",
        "run_a32_unknown_game_parameterized_control_revision_consumer",
        "write_a32_unknown_game_parameterized_control_revision_decisions",
    }
    second_unknown_game_control_dependence_revision_names = {
        "A32SecondUnknownGameControlDependenceRevisionDecision",
        "DEFAULT_A32_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_REVISION_OUTPUT_PATH",
        "CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_CONTRAST",
        "KEEP_STANDALONE_ACTION2_EFFECT_UNRESOLVED_NON_IDENTIFIABLE",
        "REFUTE_CONTROL_DEPENDENT_CONTRAST",
        "REQUEST_MORE_TESTS_FOR_CONTROL_DEPENDENCE",
        "build_a32_second_unknown_game_control_dependence_revision_decisions",
        "run_a32_second_unknown_game_control_dependence_revision_consumer",
        "write_a32_second_unknown_game_control_dependence_revision_decisions",
    }
    third_unknown_game_parameterized_relation_revision_names = {
        "A32ThirdUnknownGameParameterizedRelationDecision",
        "DEFAULT_A32_THIRD_UNKNOWN_GAME_PARAMETERIZED_RELATION_REVISION_PATH",
        "CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_PARAMETERIZED_RELATION",
        "KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_TARGET_EFFECT",
        "REFUTE_CONTROL_DEPENDENT_PARAMETERIZED_RELATION",
        "REQUEST_MORE_TESTS_FOR_PARAMETERIZED_RELATION",
        "build_a32_third_unknown_game_parameterized_relation_decisions",
        "run_a32_third_unknown_game_parameterized_relation_revision",
        "write_a32_third_unknown_game_parameterized_relation_revision",
    }
    if name in third_unknown_game_parameterized_relation_revision_names:
        import importlib

        module = importlib.import_module(
            ".third_unknown_game_parameterized_relation_revision_decisions",
            __name__,
        )
        return getattr(module, name)
    if name in second_unknown_game_control_dependence_revision_names:
        import importlib

        module = importlib.import_module(
            ".second_unknown_game_control_dependence_revision_decisions",
            __name__,
        )
        return getattr(module, name)
    if name in unknown_game_parameterized_revision_names:
        import importlib

        module = importlib.import_module(
            ".unknown_game_parameterized_control_revision_decisions",
            __name__,
        )
        return getattr(module, name)
    if name in unknown_game_protocol_names:
        import importlib

        module = importlib.import_module(
            ".unknown_game_control_protocol_decisions",
            __name__,
        )
        return getattr(module, name)
    if name in patch_decision_names:
        import importlib

        module = importlib.import_module(
            ".patch_similarity_revision_decisions", __name__
        )
        return getattr(module, name)
    if name in patch_intake_names:
        import importlib

        module = importlib.import_module(".patch_similarity_revision_intake", __name__)
        return getattr(module, name)
    decision_names = (
        set(__all__)
        - patch_intake_names
        - patch_decision_names
        - unknown_game_protocol_names
        - unknown_game_parameterized_revision_names
        - second_unknown_game_control_dependence_revision_names
        - third_unknown_game_parameterized_relation_revision_names
    )
    if name in decision_names:
        import importlib

        module = importlib.import_module(".revision_decisions", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
