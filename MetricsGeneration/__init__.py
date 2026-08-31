"""Feature engineering: per-symbol indicators plus cross-sectional panel features."""

from __future__ import annotations

from Common.io import available_metrics, matrix_path, read_matrix, write_matrix

from .indicators import (
    LABEL_COLUMNS,
    MIN_PERIODS,
    TRADING_DAYS,
    compute_labels,
    compute_metrics,
    metric_dtype,
    metric_names,
    min_periods,
)
from .metrics_generation import (
    MANIFEST_FILENAME,
    assemble_metric,
    build,
    read_manifest,
    resolve_symbols,
)
from .storage import load_bars

__all__ = [
    "LABEL_COLUMNS",
    "MANIFEST_FILENAME",
    "MIN_PERIODS",
    "TRADING_DAYS",
    "assemble_metric",
    "available_metrics",
    "build",
    "compute_labels",
    "compute_metrics",
    "load_bars",
    "matrix_path",
    "metric_dtype",
    "metric_names",
    "min_periods",
    "read_manifest",
    "read_matrix",
    "resolve_symbols",
    "write_matrix",
]
