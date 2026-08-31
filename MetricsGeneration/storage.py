"""Parquet storage helpers for the wide metric matrices (index = date, columns = symbols)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


COMPRESSION = "zstd"
INDEX_NAME = "date"
FLOAT_ENCODING = "BYTE_STREAM_SPLIT"

_CSV_RENAMES = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}
_REPAIRED_CSV_COLUMN = "Repaired?"


def load_ohlcv(path: Path) -> pd.DataFrame:
    """Read one cleaned OHLCV CSV into a lowercase frame with a sorted UTC DatetimeIndex."""
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
    result["repaired"] = repaired.astype("float64")
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def write_matrix(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.index.name = INDEX_NAME
    table = pa.Table.from_pandas(frame)
    # Byte-stream-split splits float bytes into planes so zstd can find structure: ~38% smaller.
    encoding = {
        name: FLOAT_ENCODING for name in table.schema.names if name != INDEX_NAME
    }
    pq.write_table(
        table,
        path,
        compression=COMPRESSION,
        use_dictionary=False,
        column_encoding=encoding,
    )


def read_matrix(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow", columns=columns)


def matrix_path(directory: Path, metric: str) -> Path:
    return directory / f"{metric}.parquet"
