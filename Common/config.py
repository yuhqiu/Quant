"""Layered settings: defaults -> config.toml -> environment -> explicit overrides.

One resolved :class:`Settings` object is the single source of truth for every path
and default in the project. Nothing else may compute a data path by hand.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

CONFIG_FILENAME = "config.toml"
ENV_PREFIX = "QUANT_"

_PATH_FIELDS = frozenset({"project_root", "data_root", "metrics_root", "results_root"})


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_root: Path = PROJECT_ROOT / "DataSource"
    metrics_root: Path = PROJECT_ROOT / "Metrics"
    results_root: Path = PROJECT_ROOT / "Results"

    provider: str = "yahoo"
    region: str = "US"
    asset_class: str = "stock"
    interval: str = "1d"

    compression: str = "zstd"
    log_level: str = "INFO"
    log_json: bool = True
    seed: int = 0

    # --- derived paths -----------------------------------------------------
    @property
    def lake_root(self) -> Path:
        return self.data_root / "lake"

    @property
    def bars_root(self) -> Path:
        return self.lake_root / "bars"

    @property
    def reference_root(self) -> Path:
        return self.lake_root / "reference"

    @property
    def report_root(self) -> Path:
        return self.lake_root / "reports"

    @property
    def log_dir(self) -> Path:
        return self.lake_root / "logs"

    @property
    def catalog_path(self) -> Path:
        return self.lake_root / "catalog.duckdb"

    @property
    def signals_root(self) -> Path:
        return self.results_root / "signals"

    @property
    def backtests_root(self) -> Path:
        return self.results_root / "backtests"

    def partition(
        self,
        region: str | None = None,
        asset_class: str | None = None,
        interval: str | None = None,
    ):
        from .types import Partition

        return Partition(
            region=region or self.region,
            asset_class=asset_class or self.asset_class,
            interval=interval or self.interval,
        )


_FIELD_NAMES = frozenset(field.name for field in fields(Settings))
_cached: Settings | None = None


def _coerce(name: str, value: Any) -> Any:
    if name in _PATH_FIELDS:
        return Path(str(value)).expanduser().resolve()
    if name == "seed":
        return int(value)
    if name == "log_json":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    return str(value)


def _from_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    section = document.get("quant", document)
    return {key: value for key, value in section.items() if key in _FIELD_NAMES}


def _from_env() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in _FIELD_NAMES:
        raw = os.environ.get(f"{ENV_PREFIX}{name.upper()}")
        if raw is not None and raw != "":
            values[name] = raw
    return values


def load_settings(config_path: Path | str | None = None, **overrides: Any) -> Settings:
    """Resolve settings without touching the process-wide cache."""
    path = Path(config_path) if config_path else Path(
        os.environ.get(f"{ENV_PREFIX}CONFIG") or PROJECT_ROOT / CONFIG_FILENAME
    )
    merged: dict[str, Any] = {}
    merged.update(_from_file(path))
    merged.update(_from_env())
    merged.update({k: v for k, v in overrides.items() if v is not None})

    unknown = set(merged) - _FIELD_NAMES
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")

    return Settings(**{name: _coerce(name, value) for name, value in merged.items()})


def settings() -> Settings:
    """The process-wide resolved settings, computed once."""
    global _cached
    if _cached is None:
        _cached = load_settings()
    return _cached


def configure(**overrides: Any) -> Settings:
    """Override settings for the current process. Intended for CLIs and tests."""
    global _cached
    _cached = replace(settings(), **{
        name: _coerce(name, value) for name, value in overrides.items() if value is not None
    })
    return _cached


def reset_settings() -> None:
    """Forget the cache so the next :func:`settings` call re-reads the environment."""
    global _cached
    _cached = None
