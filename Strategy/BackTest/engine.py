"""Market data bundle and execution assumptions shared by both engines.

Everything the simulation needs is materialised once as dense float arrays, so the
engines differ only in how they decide, never in what they can see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

FillPrice = Literal["next_open", "close", "vwap_proxy"]


def _utc(value: pd.Timestamp | str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


@dataclass(frozen=True)
class ExecutionConfig:
    """How orders become fills. Every assumption here is a modelling choice."""

    fill_price: FillPrice = "next_open"
    execution_lag: int = 1
    participation_rate: float = 0.1
    carry_unfilled: bool = False
    initial_capital: float = 1_000_000.0
    allow_margin: bool = False
    whole_shares: bool = True

    def __post_init__(self) -> None:
        if self.execution_lag < 0:
            raise ValueError("execution_lag cannot be negative")
        if not 0.0 < self.participation_rate <= 1.0:
            raise ValueError("participation_rate must be in (0, 1]")


@dataclass
class MarketData:
    """Dense OHLCV plus corporate actions and the inputs the cost model needs."""

    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    dividend: np.ndarray
    split_ratio: np.ndarray
    sigma: np.ndarray
    advd: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.dates), len(self.symbols)

    def fill_prices(self, mode: FillPrice) -> np.ndarray:
        if mode == "next_open":
            return self.open
        if mode == "close":
            return self.close
        if mode == "vwap_proxy":
            return (self.high + self.low + self.close) / 3.0
        raise ValueError(f"unknown fill price mode {mode!r}")

    @classmethod
    def from_frames(
        cls,
        frames: dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
        symbols: list[str] | tuple[str, ...],
    ) -> MarketData:
        columns = list(symbols)

        def array(name: str, default: float = np.nan) -> np.ndarray:
            frame = frames.get(name)
            if frame is None:
                return np.full((len(dates), len(columns)), default)
            aligned = frame.reindex(index=dates, columns=columns)
            return aligned.to_numpy(dtype="float64")

        return cls(
            dates=pd.DatetimeIndex(dates, name="date"),
            symbols=tuple(columns),
            open=array("open"),
            high=array("high"),
            low=array("low"),
            close=array("close"),
            volume=array("volume", 0.0),
            dividend=np.nan_to_num(array("dividend", 0.0)),
            split_ratio=np.nan_to_num(array("split_ratio", 0.0)),
            sigma=array("vol_20d"),
            advd=array("advd_20", 0.0),
        )

    @classmethod
    def from_panel(
        cls,
        panel,
        symbols: list[str] | tuple[str, ...] | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> MarketData:
        needed = ("open", "high", "low", "close", "volume", "dividend", "split_ratio", "vol_20d", "advd_20")
        frames = {name: panel.get(name) for name in needed if name in panel}

        dates = frames["close"].index
        if start is not None:
            dates = dates[dates >= _utc(start)]
        if end is not None:
            dates = dates[dates <= _utc(end)]
        columns = list(symbols) if symbols else list(frames["close"].columns)
        return cls.from_frames(frames, dates, columns)


@dataclass
class TradeLog:
    """Accumulates fills as columnar chunks; one concat at the end, not one per bar."""

    rows: list[dict[str, np.ndarray | pd.Timestamp]] = field(default_factory=list)

    def add(
        self,
        date: pd.Timestamp,
        symbols: np.ndarray,
        qty: np.ndarray,
        price: np.ndarray,
        commission: np.ndarray,
        spread: np.ndarray,
        slippage: np.ndarray,
    ) -> None:
        if qty.size == 0:
            return
        self.rows.append(
            {
                "date": np.full(qty.shape, date),
                "symbol": symbols,
                "side": np.where(qty > 0.0, "buy", "sell"),
                "qty": qty,
                "price": price,
                "notional": np.abs(qty) * price,
                "commission": commission,
                "spread": spread,
                "slippage": slippage,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        from .result import TRADE_COLUMNS, _empty_trades

        if not self.rows:
            return _empty_trades()
        merged = {
            key: np.concatenate([chunk[key] for chunk in self.rows])
            for key in self.rows[0]
        }
        frame = pd.DataFrame(merged)[list(TRADE_COLUMNS)]
        frame["date"] = pd.DatetimeIndex(frame["date"])
        return frame.reset_index(drop=True)


def positions_frame(
    dates: pd.DatetimeIndex,
    symbols: tuple[str, ...],
    shares: np.ndarray,
    values: np.ndarray,
    equity: np.ndarray,
) -> pd.DataFrame:
    """Long-format holdings: sparse by nature, so long beats wide on disk."""
    rows, columns = np.nonzero(shares)
    if rows.size == 0:
        return pd.DataFrame(
            {
                "date": pd.DatetimeIndex([], tz="UTC"),
                "symbol": pd.Series(dtype="object"),
                "shares": pd.Series(dtype="float64"),
                "value": pd.Series(dtype="float64"),
                "weight": pd.Series(dtype="float64"),
            }
        )

    scale = np.where(equity[rows] != 0.0, equity[rows], np.nan)
    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates[rows]),
            "symbol": np.asarray(symbols, dtype=object)[columns],
            "shares": shares[rows, columns],
            "value": values[rows, columns],
            "weight": values[rows, columns] / scale,
        }
    )


def lagged_targets(
    targets: pd.DataFrame, dates: pd.DatetimeIndex, symbols: tuple[str, ...], lag: int
) -> np.ndarray:
    """Align target weights to the bar they execute on. This is the lookahead guard."""
    aligned = targets.reindex(index=dates, columns=list(symbols))
    if lag:
        aligned = aligned.shift(lag)
    return aligned.to_numpy(dtype="float64")
