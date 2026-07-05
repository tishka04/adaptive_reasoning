"""Prefix compression for V4."""

from __future__ import annotations

from ..schemas import PrimitiveAction


class PrefixCompressor:
    """Compress successful traces into shorter useful prefixes."""

    def compress(
        self,
        successful_trace: list[PrimitiveAction],
        replay_fn=None,
    ) -> list[PrimitiveAction]:
        if not successful_trace:
            return []

        compressed: list[PrimitiveAction] = []
        for action in successful_trace:
            if compressed and action == compressed[-1]:
                continue
            compressed.append(action)

        if replay_fn is None:
            return compressed

        best = compressed[:]
        for index in range(len(compressed) - 1, -1, -1):
            candidate = compressed[:index] + compressed[index + 1 :]
            if candidate and replay_fn(candidate):
                best = candidate
        return best
