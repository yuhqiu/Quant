"""NYSE/Nasdaq trading calendar: sessions, holidays and early closes.

Rule-based rather than table-based, so it extends into the future without a data
file, plus an explicit list of one-off closures that no rule can predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

EXCHANGE_TZ = "America/New_York"
REGULAR_CLOSE = pd.Timedelta(hours=16)
EARLY_CLOSE = pd.Timedelta(hours=13)
MARKET_OPEN = pd.Timedelta(hours=9, minutes=30)

# Closures no rule predicts: presidential funerals, 9/11, hurricane Sandy.
AD_HOC_CLOSURES: frozenset[date] = frozenset(
    {
        date(1994, 4, 27),   # Nixon funeral
        date(2001, 9, 11),
        date(2001, 9, 12),
        date(2001, 9, 13),
        date(2001, 9, 14),
        date(2004, 6, 11),   # Reagan funeral
        date(2007, 1, 2),    # Ford funeral
        date(2012, 10, 29),  # Hurricane Sandy
        date(2012, 10, 30),
        date(2018, 12, 5),   # G. H. W. Bush funeral
        date(2025, 1, 9),    # Carter funeral
    }
)


def easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    offset = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * offset) // 451
    month, day = divmod(h + offset - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """``n``-th ``weekday`` (Mon=0) of a month; ``n = -1`` means the last one."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    next_month = date(year + month // 12, month % 12 + 1, 1)
    last = next_month - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date | None:
    """Saturday holidays fall back to Friday, Sunday holidays roll to Monday."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def holidays(year: int) -> set[date]:
    """Full-day NYSE closures in one calendar year."""
    result: set[date] = set()

    new_year = date(year, 1, 1)
    # A Saturday New Year's Day is not observed on the preceding Friday.
    if new_year.weekday() != 5:
        result.add(_observed(new_year) or new_year)

    if year >= 1998:
        result.add(_nth_weekday(year, 1, 0, 3))          # Martin Luther King Jr. Day
    result.add(_nth_weekday(year, 2, 0, 3))              # Washington's Birthday
    result.add(easter(year) - timedelta(days=2))         # Good Friday
    result.add(_nth_weekday(year, 5, 0, -1))             # Memorial Day
    if year >= 2022:
        result.add(_observed(date(year, 6, 19)))         # Juneteenth
    result.add(_observed(date(year, 7, 4)))              # Independence Day
    result.add(_nth_weekday(year, 9, 0, 1))              # Labor Day
    result.add(_nth_weekday(year, 11, 3, 4))             # Thanksgiving
    result.add(_observed(date(year, 12, 25)))            # Christmas

    result.update(day for day in AD_HOC_CLOSURES if day.year == year)
    return {day for day in result if day is not None and day.weekday() < 5}


def early_close_days(year: int) -> set[date]:
    """Half sessions (13:00 close)."""
    closed = holidays(year)
    candidates = {
        date(year, 7, 3),                                 # day before Independence Day
        _nth_weekday(year, 11, 3, 4) + timedelta(days=1),  # day after Thanksgiving
        date(year, 12, 24),                               # Christmas Eve
    }
    return {day for day in candidates if day.weekday() < 5 and day not in closed}


@dataclass(frozen=True, slots=True)
class TradingCalendar:
    """Sessions for one exchange. ``name`` is informational; rules are XNYS."""

    name: str = "XNYS"

    def sessions(
        self, start: pd.Timestamp | str, end: pd.Timestamp | str
    ) -> pd.DatetimeIndex:
        """Trading days in ``[start, end]`` as tz-aware UTC midnight timestamps."""
        first = pd.Timestamp(start).tz_localize(None) if pd.Timestamp(start).tzinfo else pd.Timestamp(start)
        last = pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tzinfo else pd.Timestamp(end)
        first, last = first.normalize(), last.normalize()
        if last < first:
            return pd.DatetimeIndex([], tz="UTC", name="date")

        closed = _closed_days(first.year, last.year)
        days = pd.date_range(first, last, freq="B")
        keep = [day for day in days if day.date() not in closed]
        return pd.DatetimeIndex(keep, name="date").tz_localize("UTC")

    def is_session(self, day: pd.Timestamp | str) -> bool:
        stamp = pd.Timestamp(day)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("UTC").tz_localize(None)
        return stamp.weekday() < 5 and stamp.date() not in _closed_days(
            stamp.year, stamp.year
        )

    def next_session(self, day: pd.Timestamp | str) -> pd.Timestamp:
        stamp = _naive(day) + pd.Timedelta(days=1)
        while not self.is_session(stamp):
            stamp += pd.Timedelta(days=1)
        return stamp.tz_localize("UTC")

    def previous_session(self, day: pd.Timestamp | str) -> pd.Timestamp:
        stamp = _naive(day) - pd.Timedelta(days=1)
        while not self.is_session(stamp):
            stamp -= pd.Timedelta(days=1)
        return stamp.tz_localize("UTC")

    def is_early_close(self, day: pd.Timestamp | str) -> bool:
        stamp = _naive(day)
        return stamp.date() in early_close_days(stamp.year)

    def close_time(self, day: pd.Timestamp | str) -> pd.Timestamp:
        """Session close as a tz-aware exchange-local timestamp."""
        stamp = _naive(day)
        offset = EARLY_CLOSE if self.is_early_close(stamp) else REGULAR_CLOSE
        return (stamp + offset).tz_localize(EXCHANGE_TZ)

    def missing_sessions(
        self,
        observed: pd.DatetimeIndex,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DatetimeIndex:
        """Sessions the exchange held that ``observed`` does not contain."""
        if len(observed) == 0:
            if start is None or end is None:
                return pd.DatetimeIndex([], tz="UTC", name="date")
            return self.sessions(start, end)

        index = pd.DatetimeIndex(observed)
        index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
        index = index.normalize()
        expected = self.sessions(start or index.min(), end or index.max())
        return expected.difference(index)

    def resample_rule(self, frequency: str) -> str:
        return {"daily": "D", "weekly": "W-FRI", "monthly": "ME", "quarterly": "QE"}[
            frequency
        ]

    def rebalance_dates(
        self, sessions: pd.DatetimeIndex, frequency: str
    ) -> pd.DatetimeIndex:
        """Last session of each period; ``daily`` returns every session."""
        index = pd.DatetimeIndex(sessions)
        if len(index) == 0:
            return index
        if frequency == "daily":
            return index
        grouped = pd.Series(index, index=index).resample(
            self.resample_rule(frequency)
        ).last()
        return pd.DatetimeIndex(grouped.dropna().to_numpy(), name=index.name)


def _naive(day: pd.Timestamp | str) -> pd.Timestamp:
    stamp = pd.Timestamp(day)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


@lru_cache(maxsize=8)
def _closed_days(first_year: int, last_year: int) -> frozenset[date]:
    days: set[date] = set()
    for year in range(first_year, last_year + 1):
        days |= holidays(year)
    return frozenset(days)


@lru_cache(maxsize=1)
def default_calendar() -> TradingCalendar:
    return TradingCalendar()
