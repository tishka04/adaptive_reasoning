"""A38 rollout-aware scope refinement package."""

__all__ = [
    "DEFAULT_A38_SCOPE_REFINEMENT_OUTPUT_PATH",
    "RolloutAwareScopeRefinement",
    "build_rollout_aware_scope_refinement",
    "run_rollout_aware_scope_refinement",
    "write_rollout_aware_scope_refinement",
]


def __getattr__(name: str):
    if name in set(__all__):
        import importlib

        module = importlib.import_module(".rollout_aware_scope_refinement", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
