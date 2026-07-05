"""A39 precondition-aware policy rollout package."""

__all__ = [
    "DEFAULT_A39_POLICY_ROLLOUT_OUTPUT_PATH",
    "PreconditionAwarePolicyRolloutStep",
    "UsagePreconditionCheck",
    "run_precondition_aware_policy_rollout",
    "write_precondition_aware_policy_rollout",
]


def __getattr__(name: str):
    if name in set(__all__):
        import importlib

        module = importlib.import_module(".precondition_aware_policy_rollout", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
