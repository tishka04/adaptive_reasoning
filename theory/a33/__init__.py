"""A33 confirmed mechanic registry package."""

__all__ = [
    "ConfirmedMechanicRegistryEntry",
    "ScopedUnknownGameRegistryEntry",
    "ControlDependentRelationalRegistryEntry",
    "ParameterizedRelationalRegistryEntry",
    "DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH",
    "DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH",
    "DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH",
    "DEFAULT_A33_PARAMETERIZED_RELATIONAL_REGISTRY_OUTPUT_PATH",
    "build_confirmed_mechanics_registry",
    "build_scoped_unknown_game_registry",
    "build_control_dependent_relational_registry",
    "build_parameterized_relational_registry",
    "run_confirmed_mechanics_registry_generation",
    "run_scoped_unknown_game_registry_generation",
    "run_control_dependent_relational_registry_generation",
    "run_parameterized_relational_registry_generation",
    "write_confirmed_mechanics_registry",
    "write_scoped_unknown_game_registry",
    "write_control_dependent_relational_registry",
    "write_parameterized_relational_registry",
]


def __getattr__(name: str):
    parameterized_relational_names = {
        "ParameterizedRelationalRegistryEntry",
        "DEFAULT_A33_PARAMETERIZED_RELATIONAL_REGISTRY_OUTPUT_PATH",
        "build_parameterized_relational_registry",
        "run_parameterized_relational_registry_generation",
        "write_parameterized_relational_registry",
    }
    if name in parameterized_relational_names:
        import importlib

        module = importlib.import_module(".parameterized_relational_registry", __name__)
        return getattr(module, name)
    relational_names = {
        "ControlDependentRelationalRegistryEntry",
        "DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH",
        "build_control_dependent_relational_registry",
        "run_control_dependent_relational_registry_generation",
        "write_control_dependent_relational_registry",
    }
    if name in relational_names:
        import importlib

        module = importlib.import_module(
            ".control_dependent_relational_registry", __name__
        )
        return getattr(module, name)
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
    if (
        name
        in set(__all__)
        - scoped_names
        - relational_names
        - parameterized_relational_names
    ):
        import importlib

        module = importlib.import_module(".confirmed_mechanics_registry", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
