"""One-off migration of the legacy one-CSV-per-symbol layout into the parquet lake.

Legacy files were downloaded with ``auto_adjust=True``, so their OHLC is already
adjusted; the migration records ``adj_close = close`` and ``adj_factor = 1`` to
keep that fact explicit rather than silently mixing conventions.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from . import catalog, lake
from .cleaning import clean_bars
from .lake import Partition
from .schema import TIMESTAMP, normalize_bars


def migrate_csv_directory(
    source: str | Path,
    partition: Partition,
    provider: str = "yahoo",
    drop_zero_volume: bool = False,
    on_progress: Callable[[int, int, str, str], None] | None = None,
) -> dict[str, int]:
    """Convert every ``*.csv`` in ``source`` into one parquet file per symbol."""
    directory = Path(source)
    files = sorted(path for path in directory.glob("*.csv") if not path.name.startswith("quality_"))
    if not files:
        raise FileNotFoundError(f"no CSV files under {directory}")

    now = pd.Timestamp.now(tz="UTC")
    counts = {"converted": 0, "skipped": 0, "rows": 0}
    state_rows: list[dict[str, object]] = []

    for index, path in enumerate(files, start=1):
        symbol = path.stem.split("_")[0].upper()
        try:
            frame = clean_bars(
                normalize_bars(pd.read_csv(path), symbol), drop_zero_volume=drop_zero_volume
            )
            if frame.empty:
                raise ValueError("no rows left after cleaning")

            lake.write_symbol(frame, symbol, partition)
            counts["converted"] += 1
            counts["rows"] += len(frame)
            state_rows.append(
                {
                    "provider": provider,
                    "region": partition.region,
                    "asset_class": partition.asset_class,
                    "interval": partition.interval,
                    "symbol": symbol,
                    "first_ts": frame[TIMESTAMP].min(),
                    "last_ts": frame[TIMESTAMP].max(),
                    "row_count": len(frame),
                    "status": "ok",
                    "message": "migrated from csv",
                    "updated_at": now,
                }
            )
            message = f"{len(frame)} rows"
        except Exception as exc:
            counts["skipped"] += 1
            message = f"SKIPPED - {type(exc).__name__}: {exc}"

        if on_progress:
            on_progress(index, len(files), symbol, message)

    catalog.record_states(state_rows)
    return counts
