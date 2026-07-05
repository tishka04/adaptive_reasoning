"""A33 confirmed mechanic registry package."""

__all__ = [
    "ConfirmedMechanicRegistryEntry",
    "DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH",
    "build_confirmed_mechanics_registry",
    "run_confirmed_mechanics_registry_generation",
    "write_confirmed_mechanics_registry",
]


def __getattr__(name: str):
    if name in set(__all__):
        import importlib

        module = importlib.import_module(".confirmed_mechanics_registry", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
