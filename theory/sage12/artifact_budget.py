"""Bounded, auditable storage for SAGE12 experiments.

The guard records a metadata inventory before and after every command, rejects
oversized derived files, and cleans only the unique scratch directory that it
created for the command.  It deliberately does not hash the multi-gigabyte Git
object database; the inventory checksum covers relative paths and sizes, while
new experiment outputs receive content hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

GIB = 1024**3
MIB = 1024**2


class StorageBudgetError(RuntimeError):
    """Raised before an experiment can exceed its frozen storage budget."""


@dataclass(frozen=True)
class BudgetLimits:
    maximum_scratch_bytes: int = 5 * GIB
    maximum_local_cache_bytes: int = 5 * GIB
    maximum_derived_file_bytes: int = 512 * MIB
    maximum_repository_bytes: int = 12 * GIB
    minimum_free_bytes: int = 100 * GIB


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    ]


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def inventory(repo_root: str | Path, scratch_dir: str | Path) -> dict[str, Any]:
    """Return a compact path/size inventory without materializing file data."""

    root = Path(repo_root).resolve()
    scratch = Path(scratch_dir).resolve()
    entries: list[tuple[str, int]] = []
    worktree_bytes = 0
    git_bytes = 0
    scratch_bytes = 0
    cache_bytes = 0
    largest_path = ""
    largest_bytes = 0
    for path in _walk_files(root):
        size = path.stat().st_size
        relative = _relative(path, root)
        entries.append((relative, size))
        if relative == ".git" or relative.startswith(".git/"):
            git_bytes += size
        else:
            worktree_bytes += size
        if path == scratch or scratch in path.parents:
            scratch_bytes += size
        if relative == ".sage12_cache" or relative.startswith(".sage12_cache/"):
            cache_bytes += size
        if size > largest_bytes:
            largest_path = relative
            largest_bytes = size
    total_bytes = worktree_bytes + git_bytes
    free_bytes = shutil.disk_usage(root).free
    encoded_entries = _canonical(entries).encode("utf-8")
    return {
        "repo_root": root.as_posix(),
        "files": len(entries),
        "worktree_bytes": worktree_bytes,
        "git_bytes": git_bytes,
        "repository_bytes": total_bytes,
        "scratch_bytes": scratch_bytes,
        "local_cache_bytes": cache_bytes,
        "free_bytes": free_bytes,
        "largest_file": {
            "path": largest_path,
            "bytes": largest_bytes,
        },
        "inventory_checksum": hashlib.sha256(encoded_entries).hexdigest(),
        "_entries": dict(entries),
    }


def _public_inventory(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if key != "_entries"}


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical(payload))
        handle.write("\n")


class StorageGuard:
    """Context manager enforcing one command's storage budget."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        output_dir: str | Path,
        command: str,
        limits: BudgetLimits | None = None,
        scratch_namespace: str = "v4_18",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        output = Path(output_dir)
        self.output_dir = (
            output.resolve()
            if output.is_absolute()
            else (self.repo_root / output).resolve()
        )
        self.command = str(command)
        self.limits = limits or BudgetLimits()
        unique = f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:10]}"
        self.scratch_dir = (
            self.repo_root
            / ".sage12_scratch"
            / scratch_namespace
            / self.command
            / unique
        )
        self.pre: dict[str, Any] | None = None
        self.started_at = 0.0

    def _check_snapshot(self, snapshot: dict[str, Any]) -> None:
        if snapshot["scratch_bytes"] > self.limits.maximum_scratch_bytes:
            raise StorageBudgetError(
                "scratch budget exceeded: "
                f"{snapshot['scratch_bytes']} > "
                f"{self.limits.maximum_scratch_bytes}"
            )
        if snapshot["local_cache_bytes"] > self.limits.maximum_local_cache_bytes:
            raise StorageBudgetError(
                "local cache budget exceeded: "
                f"{snapshot['local_cache_bytes']} > "
                f"{self.limits.maximum_local_cache_bytes}"
            )
        if snapshot["repository_bytes"] > self.limits.maximum_repository_bytes:
            raise StorageBudgetError(
                "repository budget exceeded: "
                f"{snapshot['repository_bytes']} > "
                f"{self.limits.maximum_repository_bytes}"
            )
        if snapshot["free_bytes"] < self.limits.minimum_free_bytes:
            raise StorageBudgetError(
                "free-space floor crossed: "
                f"{snapshot['free_bytes']} < {self.limits.minimum_free_bytes}"
            )

    def __enter__(self) -> Self:
        self.scratch_dir.mkdir(parents=True, exist_ok=False)
        self.started_at = time.time()
        self.pre = inventory(self.repo_root, self.scratch_dir)
        try:
            self._check_snapshot(self.pre)
        except Exception:
            shutil.rmtree(self.scratch_dir, ignore_errors=True)
            raise
        return self

    def _output_artifacts(self) -> list[dict[str, Any]]:
        artifacts = []
        for path in sorted(_walk_files(self.output_dir)):
            size = path.stat().st_size
            artifacts.append(
                {
                    "path": _relative(path, self.repo_root),
                    "bytes": size,
                    "sha256": _file_sha256(path),
                }
            )
        return artifacts

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        failure: Exception | None = None
        post_before_cleanup = inventory(self.repo_root, self.scratch_dir)
        try:
            self._check_snapshot(post_before_cleanup)
            previous = (self.pre or {}).get("_entries", {})
            for relative, size in post_before_cleanup["_entries"].items():
                if size <= self.limits.maximum_derived_file_bytes:
                    continue
                if previous.get(relative) == size:
                    continue
                raise StorageBudgetError(
                    f"derived file exceeds 512 MiB: {relative} ({size} bytes)"
                )
        except StorageBudgetError as caught:
            failure = caught
        artifacts: list[dict[str, Any]] = []
        try:
            artifacts = self._output_artifacts()
        except OSError as caught:
            failure = failure or caught
        shutil.rmtree(self.scratch_dir, ignore_errors=False)
        post_after_cleanup = inventory(self.repo_root, self.scratch_dir)
        event = {
            "format_version": "sage12-storage-event-v1",
            "command": self.command,
            "started_unix": self.started_at,
            "elapsed_seconds": time.time() - self.started_at,
            "limits": asdict(self.limits),
            "pre": _public_inventory(self.pre or {}),
            "post_before_cleanup": _public_inventory(post_before_cleanup),
            "post_after_cleanup": _public_inventory(post_after_cleanup),
            "scratch_removed": not self.scratch_dir.exists(),
            "outputs": artifacts,
            "command_error": None if error is None else repr(error),
            "budget_error": None if failure is None else repr(failure),
        }
        event["event_checksum"] = hashlib.sha256(
            _canonical(event).encode("utf-8")
        ).hexdigest()
        _append_jsonl(self.output_dir / "storage_events.jsonl", event)
        if failure is not None and error is None:
            raise failure
        return False
