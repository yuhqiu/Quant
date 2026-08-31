"""Storage for the metrics stage: bars in from the lake, wide matrices out."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from Common.io import INDEX_NAME, matrix_path, read_matrix, write_matrix
from Common.types import Partition
from DataAcquisition import read_symbol

BAR_COLUMNS = (
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "adj_factor",
    "dividend",
    "split_ratio",
    "repaired",
)

_CSV_RENAMES = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}
_REPAIRED_CSV_COLUMN = "Repaired?"

__all__ = [
    "BAR_COLUMNS",
    "INDEX_NAME",
    "load_bars",
    "load_ohlcv",
    "matrix_path",
    "read_matrix",
    "write_matrix",
]


def load_bars(symbol: str, partition: Partition) -> pd.DataFrame:
    """One symbol from the parquet lake, indexed by UTC bar timestamp."""
    frame = read_symbol(symbol, partition, columns=list(BAR_COLUMNS))
    if frame.empty:
        return pd.DataFrame()

    timestamps = pd.to_datetime(frame["ts"], utc=True)
    frame = frame.drop(columns=["ts"])
    frame.index = pd.DatetimeIndex(timestamps, name=INDEX_NAME)
    frame["repaired"] = frame["repaired"].astype("float64")
    frame = frame.astype("float64")
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def load_ohlcv(path: Path) -> pd.DataFrame:
    """Read one legacy OHLCV CSV into a lowercase frame with a sorted UTC index."""
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()

    timestamps = pd.to_datetime(frame.iloc[:, 0], errors="coerce", utc=True)
    frame = frame.loc[timestamps.notna()].copy()
    frame.index = pd.DatetimeIndex(timestamps.dropna(), name=INDEX_NAME)

    if _REPAIRED_CSV_COLUMN in frame.columns:
        repaired = (
            frame[_REPAIRED_CSV_COLUMN].astype(str).str.strip().str.lower().eq("true")
        )
    else:
        repaired = pd.Series(False, index=frame.index)

    frame = frame.rename(columns=_CSV_RENAMES)
    missing = [name for name in _CSV_RENAMES.values() if name not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    result = frame[list(_CSV_RENAMES.values())].astype("float64")
    if "Adj Close" in frame.columns:
        result["adj_close"] = pd.to_numeric(frame["Adj Close"], errors="coerce")
    result["repaired"] = repaired.astype("float64")
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result
