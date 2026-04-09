"""
Structural memory — stores and retrieves reusable reasoning patterns.

Two components:
  1. EpisodicRetriever: finds similar past trajectories using vector similarity
  2. StructuralMemory: stores templates, known formulations, failure patterns,
     and latent priors over domains and solver choices

Uses FAISS for efficient nearest-neighbor retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class EpisodicRetriever:
    """
    Retrieves similar past reasoning trajectories using vector similarity
    over latent state embeddings.
    """

    def __init__(self, dim: int = 128, index_type: str = "flat"):
        self.dim = dim
        self._embeddings: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []
        self._index = None
        self._index_type = index_type

    def _build_index(self) -> None:
        """Build or rebuild the FAISS index."""
        try:
            import faiss
        except ImportError:
            self._index = None
            return

        if not self._embeddings:
            self._index = None
            return

        data = np.stack(self._embeddings).astype(np.float32)
        if self._index_type == "flat":
            self._index = faiss.IndexFlatL2(self.dim)
        else:
            self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(data)

    def add(
        self,
        embedding: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> None:
        """Add a trajectory embedding with metadata."""
        vec = embedding.detach().cpu().numpy().flatten()
        if vec.shape[0] != self.dim:
            # Pad or truncate
            padded = np.zeros(self.dim, dtype=np.float32)
            n = min(vec.shape[0], self.dim)
            padded[:n] = vec[:n]
            vec = padded
        self._embeddings.append(vec)
        self._metadata.append(metadata)
        self._index = None  # Invalidate index

    def query(
        self,
        embedding: torch.Tensor,
        k: int = 5,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Find k nearest trajectories to the given embedding.

        Returns list of (distance, metadata) tuples.
        """
        if not self._embeddings:
            return []

        if self._index is None:
            self._build_index()

        if self._index is None:
            # Fallback: brute-force numpy
            return self._brute_force_query(embedding, k)

        import faiss

        query_vec = embedding.detach().cpu().numpy().flatten().astype(np.float32)
        if query_vec.shape[0] != self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            n = min(query_vec.shape[0], self.dim)
            padded[:n] = query_vec[:n]
            query_vec = padded

        query_vec = query_vec.reshape(1, -1)
        distances, indices = self._index.search(query_vec, min(k, len(self._embeddings)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self._metadata):
                results.append((float(dist), self._metadata[idx]))
        return results

    def _brute_force_query(
        self, embedding: torch.Tensor, k: int
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Fallback query without FAISS."""
        query = embedding.detach().cpu().numpy().flatten()
        if query.shape[0] != self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            n = min(query.shape[0], self.dim)
            padded[:n] = query[:n]
            query = padded

        distances = []
        for i, emb in enumerate(self._embeddings):
            dist = float(np.sum((query - emb) ** 2))
            distances.append((dist, i))

        distances.sort()
        results = []
        for dist, idx in distances[:k]:
            results.append((dist, self._metadata[idx]))
        return results

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self._embeddings:
            np.save(path / "episodic_embeddings.npy", np.stack(self._embeddings))
        with open(path / "episodic_metadata.json", "w") as f:
            json.dump(self._metadata, f, indent=2)

    def load(self, directory: str) -> None:
        path = Path(directory)
        emb_path = path / "episodic_embeddings.npy"
        meta_path = path / "episodic_metadata.json"
        if emb_path.exists():
            data = np.load(emb_path)
            self._embeddings = [data[i] for i in range(data.shape[0])]
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)
        self._index = None


class StructuralMemory:
    """
    Stores reusable reasoning templates, known formulations,
    common failure patterns, and domain priors.

    This is a simple key-value store with tagging for retrieval.
    """

    def __init__(self):
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._failure_patterns: List[Dict[str, Any]] = []
        self._domain_priors: Dict[str, Dict[str, float]] = {}
        self._formulations: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------
    def add_template(
        self,
        name: str,
        domain: str,
        template: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> None:
        self._templates[name] = {
            "domain": domain,
            "template": template,
            "tags": tags or [],
            "use_count": 0,
            "success_count": 0,
        }

    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        return self._templates.get(name)

    def find_templates(
        self, domain: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        results = []
        for name, entry in self._templates.items():
            if domain and entry["domain"] != domain:
                continue
            if tags and not any(t in entry["tags"] for t in tags):
                continue
            results.append({"name": name, **entry})
        return results

    def record_template_use(self, name: str, success: bool) -> None:
        if name in self._templates:
            self._templates[name]["use_count"] += 1
            if success:
                self._templates[name]["success_count"] += 1

    # ------------------------------------------------------------------
    # Failure patterns
    # ------------------------------------------------------------------
    def add_failure_pattern(
        self,
        domain: str,
        mode: str,
        description: str,
        resolution: Optional[str] = None,
    ) -> None:
        self._failure_patterns.append({
            "domain": domain,
            "mode": mode,
            "description": description,
            "resolution": resolution,
            "count": 1,
        })

    def find_failure_patterns(
        self, domain: Optional[str] = None, mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        results = []
        for fp in self._failure_patterns:
            if domain and fp["domain"] != domain:
                continue
            if mode and fp["mode"] != mode:
                continue
            results.append(fp)
        return results

    # ------------------------------------------------------------------
    # Domain priors (mode → expected success probability)
    # ------------------------------------------------------------------
    def update_domain_prior(self, domain: str, mode: str, success: bool) -> None:
        if domain not in self._domain_priors:
            self._domain_priors[domain] = {}
        key = f"{mode}_success"
        key_count = f"{mode}_count"
        self._domain_priors[domain][key_count] = self._domain_priors[domain].get(key_count, 0) + 1
        if success:
            self._domain_priors[domain][key] = self._domain_priors[domain].get(key, 0) + 1

    def get_mode_prior(self, domain: str, mode: str) -> float:
        """Get estimated success probability for a mode in a domain."""
        if domain not in self._domain_priors:
            return 0.5
        count = self._domain_priors[domain].get(f"{mode}_count", 0)
        if count == 0:
            return 0.5
        successes = self._domain_priors[domain].get(f"{mode}_success", 0)
        return successes / count

    # ------------------------------------------------------------------
    # Known formulations
    # ------------------------------------------------------------------
    def add_formulation(
        self,
        name: str,
        domain: str,
        formulation: Dict[str, Any],
    ) -> None:
        self._formulations[name] = {"domain": domain, "formulation": formulation}

    def get_formulation(self, name: str) -> Optional[Dict[str, Any]]:
        return self._formulations.get(name)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "templates": self._templates,
            "failure_patterns": self._failure_patterns,
            "domain_priors": self._domain_priors,
            "formulations": self._formulations,
        }
        with open(path / "structural_memory.json", "w") as f:
            json.dump(data, f, indent=2)

    def load(self, directory: str) -> None:
        path = Path(directory) / "structural_memory.json"
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self._templates = data.get("templates", {})
        self._failure_patterns = data.get("failure_patterns", [])
        self._domain_priors = data.get("domain_priors", {})
        self._formulations = data.get("formulations", {})
