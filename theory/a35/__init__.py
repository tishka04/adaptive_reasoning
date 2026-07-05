"""A35 confirmed mechanic scope mapper package."""

__all__ = [
    "ConfirmedMechanicScopeMap",
    "ContextualMechanicScopeProbe",
    "DEFAULT_A35_SCOPE_MAP_OUTPUT_PATH",
    "build_scope_map_for_mechanic",
    "run_confirmed_mechanic_scope_map",
    "write_confirmed_mechanic_scope_map",
]


def __getattr__(name: str):
    if name in set(__all__):
        import importlib

        module = importlib.import_module(".confirmed_mechanic_scope_map", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
