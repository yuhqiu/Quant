"""Yahoo Finance provider built on yfinance."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import pandas as pd
import yfinance as yf

from .base import FetchRequest, FetchResult, MarketDataProvider


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data.columns, pd.MultiIndex):
        return data
    flattened = data.copy()
    flattened.columns = [
        "_".join(str(part) for part in column if str(part))
        for column in data.columns.to_flat_index()
    ]
    return flattened


def _extract_symbol_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """Pull a single ticker out of a possibly multi-ticker yfinance response."""
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw
    for level in range(raw.columns.nlevels):
        if symbol in raw.columns.get_level_values(level):
            return raw.xs(symbol, axis=1, level=level)
    return None


class YahooProvider(MarketDataProvider):
    name: ClassVar[str] = "yahoo"
    intervals: ClassVar[frozenset[str]] = frozenset(
        {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}
    )
    max_batch_size: ClassVar[int] = 50
    request_pause: ClassVar[float] = 1.0
    # Yahoo silently truncates intraday history; clamping avoids empty responses.
    max_lookback: ClassVar[Mapping[str, pd.Timedelta]] = {
        "1m": pd.Timedelta(days=29),
        "2m": pd.Timedelta(days=59),
        "5m": pd.Timedelta(days=59),
        "15m": pd.Timedelta(days=59),
        "30m": pd.Timedelta(days=59),
        "60m": pd.Timedelta(days=729),
        "90m": pd.Timedelta(days=59),
        "1h": pd.Timedelta(days=729),
    }

    def __init__(self, prepost: bool = False, timeout: float = 10.0, **download_kwargs: Any) -> None:
        self.download_kwargs: dict[str, Any] = {
            # Raw prices plus actions: adjustments are derived and stored, not baked in.
            "auto_adjust": False,
            "actions": True,
            "repair": True,
            "keepna": False,
            "rounding": False,
            "threads": True,
            "progress": False,
            "prepost": prepost,
            "timeout": timeout,
            **download_kwargs,
        }

    def fetch(self, request: FetchRequest) -> FetchResult:
        self.validate_interval(request.interval)
        result = FetchResult()
        if not request.symbols:
            return result

        window: dict[str, Any] = {}
        if request.start is not None:
            window["start"] = request.start.tz_convert("UTC").date().isoformat()
        if request.end is not None:
            window["end"] = request.end.tz_convert("UTC").date().isoformat()
        if not window:
            window["period"] = "max"

        try:
            raw = yf.download(
                tickers=list(request.symbols),
                interval=request.interval,
                group_by="ticker",  # required by _extract_symbol_frame
                **window,
                **self.download_kwargs,
            )
        except Exception as exc:  # one bad batch must not abort the whole run
            message = f"{type(exc).__name__}: {exc}"
            result.errors = {symbol: message for symbol in request.symbols}
            return result

        for symbol in request.symbols:
            frame = _extract_symbol_frame(raw, symbol) if raw is not None else None
            if frame is None:
                result.errors[symbol] = "symbol missing from provider response"
                continue
            frame = _flatten_columns(frame).dropna(how="all")
            if frame.empty:
                result.errors[symbol] = "no data returned"
                continue
            result.frames[symbol] = frame

        return result
