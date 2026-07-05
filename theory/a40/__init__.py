"""A40 frontier handoff request package."""

__all__ = [
    "DEFAULT_A40_FRONTIER_HANDOFF_OUTPUT_PATH",
    "FrontierHandoffRequest",
    "build_frontier_handoff_requests",
    "run_frontier_handoff_requests",
    "write_frontier_handoff_requests",
]


def __getattr__(name: str):
    if name in set(__all__):
        import importlib

        module = importlib.import_module(".frontier_handoff_requests", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
