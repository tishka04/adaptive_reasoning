"""A33 confirmed mechanic registry package."""

__all__ = [
    "ConfirmedMechanicRegistryEntry",
    "ScopedUnknownGameRegistryEntry",
    "DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH",
    "DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH",
    "build_confirmed_mechanics_registry",
    "build_scoped_unknown_game_registry",
    "run_confirmed_mechanics_registry_generation",
    "run_scoped_unknown_game_registry_generation",
    "write_confirmed_mechanics_registry",
    "write_scoped_unknown_game_registry",
]


def __getattr__(name: str):
    scoped_names = {
        "ScopedUnknownGameRegistryEntry",
        "DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH",
        "build_scoped_unknown_game_registry",
        "run_scoped_unknown_game_registry_generation",
        "write_scoped_unknown_game_registry",
    }
    if name in scoped_names:
        import importlib

        module = importlib.import_module(".scoped_unknown_game_registry", __name__)
        return getattr(module, name)
    if name in set(__all__) - scoped_names:
        import importlib

        module = importlib.import_module(".confirmed_mechanics_registry", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
