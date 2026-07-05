"""A37 scope-conditioned closed-loop policy rollout package."""

__all__ = [
    "DEFAULT_A37_POLICY_ROLLOUT_OUTPUT_PATH",
    "PolicyRolloutStep",
    "ScopeConditionedRolloutDecision",
    "run_scope_conditioned_policy_rollout",
    "write_scope_conditioned_policy_rollout",
]


def __getattr__(name: str):
    if name in set(__all__):
        import importlib

        module = importlib.import_module(".scope_conditioned_policy_rollout", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
