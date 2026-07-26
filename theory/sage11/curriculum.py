"""SAGE.10g multi-source frozen schema curriculum for SAGE.11 collection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from ..online_transferable_causal_schema import (
    FrozenCausalSchemaLibrary,
    merge_frozen_causal_schema_libraries,
)
from .splits import (
    NEURO_HOLDOUT_V1,
    SOURCE_TRAIN,
    SAGE11_SPLITS,
    short_game_id,
)


@dataclass(frozen=True)
class FrozenSchemaCurriculum:
    """One immutable library frozen before validation or target evaluation."""

    library: FrozenCausalSchemaLibrary
    source_checksums: Mapping[str, str]
    split_registry_checksum: str
    format_version: str = "sage10g-curriculum-v1"

    @classmethod
    def build(
        cls,
        source_libraries: Mapping[str, FrozenCausalSchemaLibrary],
        *,
        max_schemas: int = 64,
    ) -> "FrozenSchemaCurriculum":
        normalized = {
            short_game_id(game): library
            for game, library in source_libraries.items()
        }
        SAGE11_SPLITS.assert_authorized(
            normalized,
            purpose="train",
        )
        for game, library in normalized.items():
            unexpected = set(library.source_tags).difference({game})
            if unexpected:
                raise ValueError(
                    f"source library {game} has mismatched provenance: "
                    + ", ".join(sorted(unexpected))
                )
        merged = merge_frozen_causal_schema_libraries(
            list(normalized.values()),
            allowed_source_tags=SOURCE_TRAIN,
            forbidden_source_tags=NEURO_HOLDOUT_V1,
            max_schemas=max_schemas,
        )
        return cls(
            library=merged,
            source_checksums={
                game: library.content_checksum
                for game, library in sorted(normalized.items())
            },
            split_registry_checksum=SAGE11_SPLITS.checksum,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "format_version": self.format_version,
            "frozen": True,
            "split_registry_checksum": self.split_registry_checksum,
            "source_checksums": dict(sorted(self.source_checksums.items())),
            "source_games": sorted(self.source_checksums),
            "merged_library_checksum": self.library.content_checksum,
            "merged_schema_count": len(self.library.schemas),
            "holdout_sources_present": sorted(
                set(self.library.source_tags).intersection(
                    NEURO_HOLDOUT_V1
                )
            ),
        }
        payload["checksum"] = hashlib.sha256(json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        return payload


__all__ = ["FrozenSchemaCurriculum"]
