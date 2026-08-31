"""Canonical bar schema shared by every provider, the lake and the catalog.

Raw (unadjusted) OHLCV is stored side by side with the derived corporate-action
values, so downstream code can rebuild any adjustment convention it needs
instead of being locked into the one that happened to be used at download time.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import pyarrow as pa

TIMESTAMP: Final = "ts"
SYMBOL: Final = "symbol"

RAW_COLUMNS: Final = ("open", "high", "low", "close", "volume")
DERIVED_COLUMNS: Final = ("adj_close", "adj_factor", "dividend", "split_ratio")
FLAG_COLUMNS: Final = ("repaired",)

BAR_COLUMNS: Final = (TIMESTAMP, SYMBOL, *RAW_COLUMNS, *DERIVED_COLUMNS, *FLAG_COLUMNS)

BAR_SCHEMA: Final = pa.schema(
    [
        (TIMESTAMP, pa.timestamp("us", tz="UTC")),
        (SYMBOL, pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("adj_close", pa.float64()),
        ("adj_factor", pa.float64()),
        ("dividend", pa.float64()),
        ("split_ratio", pa.float64()),
        ("repaired", pa.bool_()),
    ]
)

# Provider column names (lower-cased, punctuation stripped) mapped to canonical ones.
_SOURCE_ALIASES: Final = {
    "date": TIMESTAMP,
    "datetime": TIMESTAMP,
    "timestamp": TIMESTAMP,
    "time": TIMESTAMP,
    "ts": TIMESTAMP,
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "closeadj": "adj_close",
    "adjclose": "adj_close",
    "adjustedclose": "adj_close",
    "volume": "volume",
    "dividends": "dividend",
    "dividend": "dividend",
    "stocksplits": "split_ratio",
    "splits": "split_ratio",
    "splitratio": "split_ratio",
    "repaired": "repaired",
}

INTERVAL_STEP: Final = {
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
    "3mo": pd.Timedelta(days=90),
}


def canonical_name(column: object) -> str | None:
    key = "".join(character for character in str(column).lower() if character.isalnum())
    return _SOURCE_ALIASES.get(key)


def empty_bars() -> pd.DataFrame:
    """An empty frame carrying the canonical columns and dtypes."""
    return BAR_SCHEMA.empty_table().to_pandas()


def interval_step(interval: str) -> pd.Timedelta:
    try:
        return INTERVAL_STEP[interval]
    except KeyError as exc:
        raise ValueError(f"unsupported interval: {interval}") from exc


def normalize_bars(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Map a provider frame onto the canonical schema, deriving what is missing.

    The index is used as the timestamp when no timestamp column is present, which
    is how most providers (yfinance included) return their data.
    """
    if frame is None or frame.empty:
        return empty_bars()

    renamed: dict[str, pd.Series] = {}
    for column in frame.columns:
        name = canonical_name(column)
        if name is not None and name not in renamed:
            renamed[name] = frame[column]

    if TIMESTAMP in renamed:
        timestamps = pd.to_datetime(renamed.pop(TIMESTAMP), errors="coerce", utc=True)
    else:
        timestamps = pd.to_datetime(pd.Series(frame.index), errors="coerce", utc=True)

    normalized = pd.DataFrame(
        {name: pd.Series(values).reset_index(drop=True) for name, values in renamed.items()}
    )
    normalized[TIMESTAMP] = pd.Series(timestamps).reset_index(drop=True)
    normalized = normalized.loc[normalized[TIMESTAMP].notna()]

    for column in (*RAW_COLUMNS, "adj_close", "dividend", "split_ratio"):
        if column in normalized:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        else:
            normalized[column] = float("nan")

    normalized["adj_close"] = normalized["adj_close"].fillna(normalized["close"])
    # Split/dividend adjusted close divided by the raw close: rebuilds adjusted OHLC.
    normalized["adj_factor"] = (normalized["adj_close"] / normalized["close"]).where(
        normalized["close"] > 0, 1.0
    )
    normalized["dividend"] = normalized["dividend"].fillna(0.0)
    normalized["split_ratio"] = normalized["split_ratio"].fillna(0.0)

    repaired = normalized["repaired"] if "repaired" in normalized else False
    normalized["repaired"] = pd.Series(repaired, index=normalized.index).astype(str).str.lower().isin(
        {"true", "1", "1.0", "yes"}
    )

    normalized[SYMBOL] = str(symbol).strip().upper()
    return normalized[list(BAR_COLUMNS)].reset_index(drop=True)
