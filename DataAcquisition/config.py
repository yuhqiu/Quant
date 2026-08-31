"""Paths and defaults for the data acquisition module.

Every location can be relocated by setting the ``QUANT_DATA_ROOT`` environment
variable, which keeps the lake out of the repository when needed.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(os.environ.get("QUANT_DATA_ROOT") or PROJECT_ROOT / "DataSource")
LAKE_ROOT = DATA_ROOT / "lake"
BARS_ROOT = LAKE_ROOT / "bars"
REFERENCE_ROOT = LAKE_ROOT / "reference"
REPORT_ROOT = LAKE_ROOT / "reports"
CATALOG_PATH = LAKE_ROOT / "catalog.duckdb"

DEFAULT_PROVIDER = "yahoo"
DEFAULT_REGION = "US"
DEFAULT_ASSET_CLASS = "stock"
DEFAULT_INTERVAL = "1d"

PARQUET_COMPRESSION = "zstd"

# Legacy one-CSV-per-symbol layout kept for the migration command.
LEGACY_CSV_ROOT = DATA_ROOT
