"""Parquet lake: one file per symbol, hive-partitioned so DuckDB can query it directly.

Layout::

    DataSource/lake/bars/region=US/asset_class=stock/interval=1d/AAPL.parquet

The partition columns live in the directory names, the symbol lives inside the
file, and writes are atomic so an interrupted run never leaves a torn file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import BARS_ROOT, PARQUET_COMPRESSION
from .schema import BAR_COLUMNS, BAR_SCHEMA, SYMBOL, TIMESTAMP, empty_bars

_UNSAFE_CHARS = re.compile(r"[^A-Z0-9._-]")
# Reserved on Windows regardless of extension; real tickers such as CON collide.
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
)


@dataclass(frozen=True)
class Partition:
    region: str
    asset_class: str
    interval: str

    @property
    def path(self) -> Path:
        return (
            BARS_ROOT
            / f"region={self.region}"
            / f"asset_class={self.asset_class}"
            / f"interval={self.interval}"
        )


def symbol_filename(symbol: str) -> str:
    stem = _UNSAFE_CHARS.sub("_", str(symbol).strip().upper())
    if stem.split(".")[0] in _RESERVED_NAMES:
        stem = f"{stem}_"
    return f"{stem}.parquet"


def filename_symbol(path: Path) -> str:
    stem = path.name[: -len(".parquet")]
    if stem.endswith("_") and stem[:-1].split(".")[0] in _RESERVED_NAMES:
        stem = stem[:-1]
    return stem


def bar_path(symbol: str, partition: Partition) -> Path:
    return partition.path / symbol_filename(symbol)


def stored_symbols(partition: Partition) -> list[str]:
    directory = partition.path
    if not directory.is_dir():
        return []
    return sorted(filename_symbol(path) for path in directory.glob("*.parquet"))


def read_symbol(symbol: str, partition: Partition, columns: list[str] | None = None) -> pd.DataFrame:
    path = bar_path(symbol, partition)
    if not path.exists():
        return empty_bars() if columns is None else empty_bars()[columns]
    return pd.read_parquet(path, engine="pyarrow", columns=columns)


def read_bars(
    partition: Partition,
    symbols: Iterable[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate the requested symbols (or the whole partition) into one long frame."""
    names = list(symbols) if symbols is not None else stored_symbols(partition)
    frames = [read_symbol(symbol, partition, columns=columns) for symbol in names]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty_bars() if columns is None else empty_bars()[columns]
    return pd.concat(frames, ignore_index=True)


def write_symbol(frame: pd.DataFrame, symbol: str, partition: Partition) -> Path:
    """Atomically replace the parquet file holding one symbol."""
    path = bar_path(symbol, partition)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = frame.reindex(columns=list(BAR_COLUMNS))
    ordered[TIMESTAMP] = pd.to_datetime(ordered[TIMESTAMP], utc=True)
    ordered[SYMBOL] = ordered[SYMBOL].astype("string").fillna(symbol.upper())
    table = pa.Table.from_pandas(ordered, schema=BAR_SCHEMA, preserve_index=False)

    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(
        table,
        temporary,
        compression=PARQUET_COMPRESSION,
        # Splitting float bytes into planes lets zstd find structure: ~38% smaller.
        column_encoding={
            name: "BYTE_STREAM_SPLIT"
            for name in table.schema.names
            if pa.types.is_floating(table.schema.field(name).type)
        },
        use_dictionary=[SYMBOL],
    )
    os.replace(temporary, path)
    return path


def last_timestamp(symbol: str, partition: Partition) -> pd.Timestamp | None:
    """Read the newest stored timestamp from parquet statistics, without loading rows."""
    path = bar_path(symbol, partition)
    if not path.exists():
        return None
    try:
        metadata = pq.ParquetFile(path).metadata
    except Exception:
        return None
    if metadata.num_rows == 0:
        return None

    column = metadata.schema.names.index(TIMESTAMP)
    newest: pd.Timestamp | None = None
    for group in range(metadata.num_row_groups):
        statistics = metadata.row_group(group).column(column).statistics
        if statistics is None or statistics.max is None:
            return _last_timestamp_from_rows(path)
        candidate = pd.Timestamp(statistics.max)
        if candidate.tzinfo is None:
            candidate = candidate.tz_localize("UTC")
        if newest is None or candidate > newest:
            newest = candidate
    return newest


def _last_timestamp_from_rows(path: Path) -> pd.Timestamp | None:
    frame = pd.read_parquet(path, engine="pyarrow", columns=[TIMESTAMP])
    if frame.empty:
        return None
    return pd.Timestamp(frame[TIMESTAMP].max())


def iter_partitions() -> Iterator[Partition]:
    """Every partition that currently holds at least one parquet file."""
    if not BARS_ROOT.is_dir():
        return
    for interval_dir in BARS_ROOT.glob("region=*/asset_class=*/interval=*"):
        if not any(interval_dir.glob("*.parquet")):
            continue
        region, asset_class, interval = (part.split("=", 1)[1] for part in interval_dir.parts[-3:])
        yield Partition(region=region, asset_class=asset_class, interval=interval)
