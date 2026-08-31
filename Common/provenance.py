"""Provenance stamps: git commit, library versions, input hashes.

A result you cannot reproduce is not a result, so every artifact carries enough
information to identify the code and the inputs that produced it.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT

_LIBRARIES = ("pandas", "numpy", "pyarrow", "duckdb", "scipy")


@lru_cache(maxsize=1)
def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


@lru_cache(maxsize=1)
def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


@lru_cache(maxsize=1)
def library_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {"python": platform.python_version()}
    for name in _LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def file_hash(path: Path | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_payload(payload: Any) -> str:
    """Stable hash of any JSON-serialisable structure."""
    import json

    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_hashes(paths: Iterable[Path | str]) -> dict[str, str]:
    """sha256 of each existing manifest / input file, keyed by name."""
    result: dict[str, str] = {}
    for item in paths:
        path = Path(item)
        if path.is_file():
            result[path.name] = file_hash(path)
    return result


def stamp(**extra: Any) -> dict[str, Any]:
    """The provenance block embedded in every artifact."""
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "versions": library_versions(),
        **extra,
    }
