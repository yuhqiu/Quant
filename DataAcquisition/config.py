"""Paths and defaults for the data acquisition module.

A lazy view over :mod:`Common.config`: the familiar constants are resolved on
every attribute access rather than frozen at import, so redirecting
``QUANT_DATA_ROOT`` after import still moves the whole module.
"""

from __future__ import annotations

from typing import Any

from Common.config import PROJECT_ROOT, settings

_DERIVED = {
    "DATA_ROOT": "data_root",
    "LAKE_ROOT": "lake_root",
    "BARS_ROOT": "bars_root",
    "REFERENCE_ROOT": "reference_root",
    "REPORT_ROOT": "report_root",
    "CATALOG_PATH": "catalog_path",
    # Legacy one-CSV-per-symbol layout kept for the migration command.
    "LEGACY_CSV_ROOT": "data_root",
    "DEFAULT_PROVIDER": "provider",
    "DEFAULT_REGION": "region",
    "DEFAULT_ASSET_CLASS": "asset_class",
    "DEFAULT_INTERVAL": "interval",
    "PARQUET_COMPRESSION": "compression",
}

__all__ = ["PROJECT_ROOT", *_DERIVED]


def __getattr__(name: str) -> Any:
    try:
        return getattr(settings(), _DERIVED[name])
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    return sorted(__all__)
