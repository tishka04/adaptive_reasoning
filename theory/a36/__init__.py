"""A36 scope-conditioned action policy package."""

__all__ = [
    "DEFAULT_A36_POLICY_PROBE_OUTPUT_PATH",
    "ScopeConditionedPolicyProbe",
    "build_scope_conditioned_policy_probe",
    "run_scope_conditioned_policy_probe",
    "write_scope_conditioned_policy_probe",
]


def __getattr__(name: str):
    if name in set(__all__):
        import importlib

        module = importlib.import_module(".scope_conditioned_policy_probe", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
