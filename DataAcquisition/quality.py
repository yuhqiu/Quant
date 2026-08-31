"""Data quality report computed in DuckDB directly over the parquet lake."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import catalog, config
from .lake import Partition

_REPORT_SQL = """
WITH scoped AS (
    SELECT * FROM read_parquet('{pattern}')
),
ordered AS (
    SELECT symbol, ts, open, high, low, close, volume, adj_factor, repaired,
           ts - lag(ts) OVER (PARTITION BY symbol ORDER BY ts) AS gap,
           close / nullif(lag(close) OVER (PARTITION BY symbol ORDER BY ts), 0) AS price_ratio
    FROM scoped
)
SELECT
    symbol,
    count(*)                                            AS rows,
    min(ts)                                             AS first_ts,
    max(ts)                                             AS last_ts,
    date_diff('day', max(ts), now())                    AS stale_days,
    count(*) FILTER (WHERE close IS NULL)               AS null_close,
    count(*) FILTER (WHERE volume IS NULL OR volume = 0) AS zero_volume,
    count(*) FILTER (WHERE repaired)                    AS repaired_rows,
    count(*) FILTER (WHERE adj_factor IS NULL OR adj_factor <= 0) AS bad_adj_factor,
    count(*) FILTER (WHERE price_ratio > 4 OR price_ratio < 0.25) AS price_jumps,
    coalesce(max(date_diff('day', ts - gap, ts)), 0)    AS max_gap_days,
    round(count(*) FILTER (WHERE repaired) / count(*), 6)               AS repaired_ratio,
    round(count(*) FILTER (WHERE volume IS NULL OR volume = 0) / count(*), 6) AS zero_volume_ratio
FROM ordered
GROUP BY symbol
ORDER BY price_jumps DESC, repaired_ratio DESC, symbol
"""


def build_report(partition: Partition) -> pd.DataFrame:
    """One row per symbol describing coverage, staleness and suspicious prices."""
    pattern = (partition.path / "*.parquet").as_posix()
    if not any(partition.path.glob("*.parquet")):
        raise FileNotFoundError(f"no parquet files under {partition.path}")
    return catalog.query(_REPORT_SQL.format(pattern=pattern))


def report_path(partition: Partition) -> Path:
    return (
        config.REPORT_ROOT
        / f"quality_{partition.region}_{partition.asset_class}_{partition.interval}.parquet"
    )


def save_report(report: pd.DataFrame, partition: Partition) -> Path:
    path = report_path(partition)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path
