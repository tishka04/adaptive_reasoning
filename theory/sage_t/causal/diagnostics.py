"""Bounded append-only diagnostics for the SAGE.T causal vertical."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DIAGNOSTIC_FILES = {
    "posterior": "posterior_trace.jsonl",
    "program": "program_registry.jsonl",
    "bundle": "intervention_bundles.jsonl",
    "prediction": "prediction_matrix.jsonl",
    "repair": "repair_lineage.jsonl",
    "memory": "memory_promotions.jsonl",
}


class CausalDiagnosticsWriter:
    def __init__(
        self,
        root: str | Path = "diagnostics/sage_t/causal",
        *,
        maximum_bytes_per_file: int = 16 * 1024 * 1024,
    ) -> None:
        self.root = Path(root)
        self.maximum_bytes_per_file = max(1024, int(maximum_bytes_per_file))
        self.errors = 0
        self.dropped_records = 0

    def write(self, channel: str, payload: Mapping[str, Any]) -> bool:
        filename = DIAGNOSTIC_FILES.get(str(channel))
        if filename is None:
            raise ValueError(f"unknown causal diagnostic channel: {channel}")
        path = self.root / filename
        encoded = json.dumps(dict(payload), sort_keys=True, default=str) + "\n"
        try:
            if path.exists() and path.stat().st_size + len(encoded.encode("utf-8")) > self.maximum_bytes_per_file:
                self.dropped_records += 1
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            return True
        except OSError:
            self.errors += 1
            return False


__all__ = ["DIAGNOSTIC_FILES", "CausalDiagnosticsWriter"]
