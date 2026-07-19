"""A34 confirmed mechanic usage probe package."""

__all__ = [
    "ConfirmedMechanicUsageProbeResult",
    "ControlDependentRelationalUsageProbeResult",
    "DEFAULT_A34_USAGE_PROBE_OUTPUT_PATH",
    "DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH",
    "build_usage_probe_for_mechanic",
    "build_a34_2_replay_contexts",
    "run_confirmed_mechanic_usage_probe",
    "run_control_dependent_relational_usage_probe",
    "write_confirmed_mechanic_usage_probe",
    "write_control_dependent_relational_usage_probe",
]


def __getattr__(name: str):
    relational_names = {
        "ControlDependentRelationalUsageProbeResult",
        "DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH",
        "build_a34_2_replay_contexts",
        "run_control_dependent_relational_usage_probe",
        "write_control_dependent_relational_usage_probe",
    }
    if name in relational_names:
        import importlib

        module = importlib.import_module(
            ".control_dependent_relational_usage_probe", __name__
        )
        return getattr(module, name)
    if name in set(__all__) - relational_names:
        import importlib

        module = importlib.import_module(".confirmed_mechanic_usage_probe", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
