"""Lazy access to the wide feature panel.

A signal declares the metrics it needs; the panel reads exactly those files and
caches them, so a three-metric signal touches three parquet files rather than the
whole build.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from Common.io import available_metrics, matrix_path, read_matrix
from Common.types import Partition

LABEL_PREFIX = "fwd_ret_"


def _utc(value) -> pd.Timestamp | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


@dataclass
class FeaturePanel:
    """Read-only view over one partition's metric matrices."""

    partition: Partition = field(default_factory=Partition)
    symbols: tuple[str, ...] | None = None
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    metrics_dir: Path | None = None
    labels_dir: Path | None = None
    _cache: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.metrics_dir = Path(self.metrics_dir or self.partition.metrics_dir)
        self.labels_dir = Path(self.labels_dir or self.partition.labels_dir)
        if self.symbols is not None:
            self.symbols = tuple(sorted({str(s).upper() for s in self.symbols}))
        self.start = _utc(self.start)
        self.end = _utc(self.end)

    # --- access ------------------------------------------------------------
    def get(self, metric: str) -> pd.DataFrame:
        """One metric as a wide date x symbol frame, cached for the panel's lifetime."""
        if metric not in self._cache:
            self._cache[metric] = self._load(metric)
        return self._cache[metric]

    def __getitem__(self, metric: str) -> pd.DataFrame:
        return self.get(metric)

    def __contains__(self, metric: str) -> bool:
        return self._path(metric).exists()

    def many(self, metrics: tuple[str, ...] | list[str]) -> dict[str, pd.DataFrame]:
        return {name: self.get(name) for name in metrics}

    def label(self, horizon: int | str) -> pd.DataFrame:
        name = horizon if isinstance(horizon, str) else f"{LABEL_PREFIX}{horizon}d"
        return self.get(name)

    def require(self, metrics: tuple[str, ...] | list[str]) -> None:
        missing = [name for name in metrics if name not in self]
        if missing:
            raise KeyError(
                f"panel {self.partition} is missing metrics {missing}; run MetricsGeneration build"
            )

    # --- metadata ----------------------------------------------------------
    @cached_property
    def available(self) -> list[str]:
        return available_metrics(self.metrics_dir) + available_metrics(self.labels_dir)

    @cached_property
    def dates(self) -> pd.DatetimeIndex:
        return self.get("close").index

    @cached_property
    def universe(self) -> list[str]:
        return list(self.get("close").columns)

    def slice(self, start=None, end=None) -> FeaturePanel:
        """A new panel over a narrower window. Caches are not shared."""
        return FeaturePanel(
            partition=self.partition,
            symbols=self.symbols,
            start=start if start is not None else self.start,
            end=end if end is not None else self.end,
            metrics_dir=self.metrics_dir,
            labels_dir=self.labels_dir,
        )

    def clear(self) -> None:
        self._cache.clear()

    # --- internals ---------------------------------------------------------
    def _path(self, metric: str) -> Path:
        primary = matrix_path(self.metrics_dir, metric)
        return primary if primary.exists() else matrix_path(self.labels_dir, metric)

    def _load(self, metric: str) -> pd.DataFrame:
        path = self._path(metric)
        if not path.exists():
            raise KeyError(f"metric {metric!r} not found under {self.metrics_dir}")
        frame = read_matrix(path, columns=list(self.symbols) if self.symbols else None)
        if self.start is not None:
            frame = frame.loc[frame.index >= self.start]
        if self.end is not None:
            frame = frame.loc[frame.index <= self.end]
        return frame.astype("float64")
