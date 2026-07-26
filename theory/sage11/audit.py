"""Print the reproducible SAGE.11 environment/split/model audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from .model import Sage11GraphWorldModel
from .splits import SAGE11_SPLITS


def build_audit() -> Dict[str, Any]:
    model = Sage11GraphWorldModel()
    return {
        "format_version": "sage11-implementation-audit-v1",
        "split_registry_checksum": SAGE11_SPLITS.checksum,
        "split_registry": SAGE11_SPLITS.to_dict(),
        "world_model": model.checkpoint_metadata(),
        "runtime": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
        },
        "authority_default": "off",
        "legacy_weights_loaded": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args(argv)
    payload = json.dumps(build_audit(), indent=2, sort_keys=True) + "\n"
    if arguments.out is None:
        print(payload, end="")
    else:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_audit", "main"]
