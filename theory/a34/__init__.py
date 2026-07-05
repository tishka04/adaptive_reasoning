"""A34 confirmed mechanic usage probe package."""

__all__ = [
    "ConfirmedMechanicUsageProbeResult",
    "DEFAULT_A34_USAGE_PROBE_OUTPUT_PATH",
    "build_usage_probe_for_mechanic",
    "run_confirmed_mechanic_usage_probe",
    "write_confirmed_mechanic_usage_probe",
]


def __getattr__(name: str):
    if name in set(__all__):
        import importlib

        module = importlib.import_module(".confirmed_mechanic_usage_probe", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
