"""Shared value objects, so no module needs to import another module's internals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import pandas as pd

Region = Literal["US"]
AssetClass = Literal["stock", "etf", "other"]
Interval = Literal[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
]

REGIONS: Final[tuple[str, ...]] = ("US",)
ASSET_CLASSES: Final[tuple[str, ...]] = ("stock", "etf", "other")
INTERVALS: Final[tuple[str, ...]] = (
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo",
)

INTERVAL_STEP: Final[dict[str, pd.Timedelta]] = {
    "1m": pd.Timedelta(minutes=1),
    "2m": pd.Timedelta(minutes=2),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "60m": pd.Timedelta(hours=1),
    "90m": pd.Timedelta(minutes=90),
    "1h": pd.Timedelta(hours=1),
    "1d": pd.Timedelta(days=1),
    "5d": pd.Timedelta(days=5),
    "1wk": pd.Timedelta(weeks=1),
    "1mo": pd.Timedelta(days=30),
    "3mo": pd.Timedelta(days=91),
}

# Bars per year, used to annualise anything measured at the bar frequency.
PERIODS_PER_YEAR: Final[dict[str, float]] = {
    "1d": 252.0,
    "5d": 50.4,
    "1wk": 52.0,
    "1mo": 12.0,
    "3mo": 4.0,
    "1h": 252.0 * 6.5,
    "60m": 252.0 * 6.5,
    "30m": 252.0 * 13.0,
    "15m": 252.0 * 26.0,
    "5m": 252.0 * 78.0,
    "1m": 252.0 * 390.0,
}

TRADING_DAYS: Final[int] = 252


def interval_step(interval: str) -> pd.Timedelta:
    try:
        return INTERVAL_STEP[interval]
    except KeyError as error:
        raise ValueError(f"unknown interval {interval!r}") from error


def periods_per_year(interval: str) -> float:
    return PERIODS_PER_YEAR.get(interval, TRADING_DAYS)


@dataclass(frozen=True, slots=True)
class Partition:
    """Addresses one dataset: a region, an asset class and a bar interval."""

    region: str = "US"
    asset_class: str = "stock"
    interval: str = "1d"

    def __post_init__(self) -> None:
        if self.asset_class not in ASSET_CLASSES:
            raise ValueError(f"unknown asset_class {self.asset_class!r}")
        if self.interval not in INTERVALS:
            raise ValueError(f"unknown interval {self.interval!r}")

    @property
    def key(self) -> str:
        return f"{self.region}/{self.asset_class}/{self.interval}"

    @property
    def hive_path(self) -> Path:
        return Path(
            f"region={self.region}",
            f"asset_class={self.asset_class}",
            f"interval={self.interval}",
        )

    @property
    def path(self) -> Path:
        """Directory holding the raw bar parquet files for this partition."""
        from .config import settings

        return settings().bars_root / self.hive_path

    @property
    def metrics_dir(self) -> Path:
        from .config import settings

        return settings().metrics_root / self.region / self.asset_class / self.interval

    @property
    def labels_dir(self) -> Path:
        from .config import settings

        return (
            settings().metrics_root
            / self.region
            / self.asset_class
            / f"labels_{self.interval}"
        )

    @property
    def periods_per_year(self) -> float:
        return periods_per_year(self.interval)

    @property
    def step(self) -> pd.Timedelta:
        return interval_step(self.interval)

    @classmethod
    def parse(cls, text: str) -> Partition:
        """``"US/stock/1d"`` -> ``Partition("US", "stock", "1d")``."""
        parts = [piece for piece in str(text).split("/") if piece]
        if len(parts) != 3:
            raise ValueError(f"expected 'region/asset_class/interval', got {text!r}")
        return cls(region=parts[0], asset_class=parts[1], interval=parts[2])

    def __str__(self) -> str:
        return self.key
