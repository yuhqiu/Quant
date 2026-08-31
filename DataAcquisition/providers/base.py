"""Provider abstraction: everything a new data vendor has to implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Mapping

import pandas as pd


@dataclass(frozen=True)
class FetchRequest:
    """One provider call: a set of symbols over a single interval and window."""

    symbols: tuple[str, ...]
    interval: str
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None


@dataclass
class FetchResult:
    """Frames keyed by symbol, plus the symbols that failed and why."""

    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


class MarketDataProvider(ABC):
    """Base class for market data vendors.

    Subclasses declare the intervals they serve, how many symbols fit in one
    request and how far back each interval may be queried; the ingest layer uses
    those declarations to build valid requests without vendor-specific logic.
    """

    name: ClassVar[str] = "base"
    intervals: ClassVar[frozenset[str]] = frozenset()
    max_batch_size: ClassVar[int] = 1
    request_pause: ClassVar[float] = 0.0
    max_lookback: ClassVar[Mapping[str, pd.Timedelta]] = {}

    @abstractmethod
    def fetch(self, request: FetchRequest) -> FetchResult:
        """Return raw provider frames keyed by symbol; never raise for one bad symbol."""

    def validate_interval(self, interval: str) -> str:
        if self.intervals and interval not in self.intervals:
            raise ValueError(
                f"{self.name} does not serve interval {interval!r}; "
                f"supported: {sorted(self.intervals)}"
            )
        return interval

    def earliest_start(self, interval: str, now: pd.Timestamp | None = None) -> pd.Timestamp | None:
        """The oldest timestamp the provider will still return for this interval."""
        lookback = self.max_lookback.get(interval)
        if lookback is None:
            return None
        reference = now if now is not None else pd.Timestamp.now(tz="UTC")
        return reference - lookback

    def clamp_start(
        self, interval: str, start: pd.Timestamp | None, now: pd.Timestamp | None = None
    ) -> pd.Timestamp | None:
        earliest = self.earliest_start(interval, now=now)
        if earliest is None:
            return start
        if start is None:
            return earliest
        return max(start, earliest)
