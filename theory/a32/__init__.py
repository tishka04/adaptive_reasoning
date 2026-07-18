"""A32 scientific revision decision package."""

__all__ = [
    "A32RevisionDecision",
    "A32PatchSimilarityRevisionIntakeCandidate",
    "A32PatchSimilarityRevisionDecision",
    "A32UnknownGameControlProtocolDecision",
    "DEFAULT_A32_REVISION_DECISIONS_OUTPUT_PATH",
    "DEFAULT_A32_PATCH_SIMILARITY_REVISION_INTAKE_OUTPUT_PATH",
    "DEFAULT_A32_PATCH_SIMILARITY_REVISION_DECISIONS_OUTPUT_PATH",
    "DEFAULT_A32_UNKNOWN_GAME_CONTROL_PROTOCOL_DECISIONS_OUTPUT_PATH",
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
    "build_a32_patch_similarity_revision_decisions",
    "build_a32_patch_similarity_revision_intake_candidates",
    "build_a32_revision_decisions",
    "build_a32_unknown_game_control_protocol_decisions",
    "run_a32_patch_similarity_revision_decision_consumer",
    "run_a32_patch_similarity_revision_intake",
    "run_a32_revision_decision_consumer",
    "run_a32_unknown_game_control_protocol_decision_consumer",
    "write_a32_patch_similarity_revision_decisions",
    "write_a32_patch_similarity_revision_intake",
    "write_a32_revision_decisions",
    "write_a32_unknown_game_control_protocol_decisions",
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
    )
    if name in decision_names:
        import importlib

        module = importlib.import_module(".revision_decisions", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
