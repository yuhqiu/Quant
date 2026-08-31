"""Market data acquisition: providers -> cleaning -> parquet lake -> DuckDB catalog."""

from __future__ import annotations

from .cleaning import clean_bars, merge_bars
from .ingest import IngestReport, ingest, resolve_symbols
from .lake import Partition, read_bars, read_symbol, stored_symbols
from .providers import MarketDataProvider, get_provider, provider_names, register_provider
from .schema import BAR_COLUMNS, BAR_SCHEMA, normalize_bars

__all__ = [
    "BAR_COLUMNS",
    "BAR_SCHEMA",
    "IngestReport",
    "MarketDataProvider",
    "Partition",
    "clean_bars",
    "get_provider",
    "ingest",
    "merge_bars",
    "normalize_bars",
    "provider_names",
    "read_bars",
    "read_symbol",
    "register_provider",
    "resolve_symbols",
    "stored_symbols",
]
