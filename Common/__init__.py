"""Shared primitives: settings, calendar, logging, parquet IO, value types."""

from __future__ import annotations

from .calendar import TradingCalendar, default_calendar
from .config import Settings, configure, load_settings, reset_settings, settings
from .io import (
    atomic_path,
    available_metrics,
    matrix_path,
    read_json,
    read_matrix,
    read_parquet,
    write_json,
    write_matrix,
    write_parquet,
)
from .logging import configure_logging, get_logger
from .provenance import file_hash, git_commit, hash_payload, library_versions, stamp
from .types import (
    ASSET_CLASSES,
    INTERVALS,
    PERIODS_PER_YEAR,
    REGIONS,
    TRADING_DAYS,
    AssetClass,
    Interval,
    Partition,
    Region,
    interval_step,
    periods_per_year,
)

__all__ = [
    "ASSET_CLASSES",
    "AssetClass",
    "INTERVALS",
    "Interval",
    "PERIODS_PER_YEAR",
    "Partition",
    "REGIONS",
    "Region",
    "Settings",
    "TRADING_DAYS",
    "TradingCalendar",
    "atomic_path",
    "available_metrics",
    "configure",
    "configure_logging",
    "default_calendar",
    "file_hash",
    "get_logger",
    "git_commit",
    "hash_payload",
    "interval_step",
    "library_versions",
    "load_settings",
    "matrix_path",
    "periods_per_year",
    "read_json",
    "read_matrix",
    "read_parquet",
    "reset_settings",
    "settings",
    "stamp",
    "write_json",
    "write_matrix",
    "write_parquet",
]
