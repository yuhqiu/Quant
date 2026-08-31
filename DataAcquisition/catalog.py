"""DuckDB catalog: incremental-download watermarks plus SQL views over the lake.

The ``ingest_state`` table is the fast path for "what do I already have?", and it
can always be rebuilt from the parquet files with :func:`refresh_from_lake`, so
the lake stays the single source of truth.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import duckdb
import pandas as pd

from .config import BARS_ROOT, CATALOG_PATH, REFERENCE_ROOT
from . import lake
from .lake import Partition

_STATE_DDL = """
CREATE TABLE IF NOT EXISTS ingest_state (
    provider     VARCHAR NOT NULL,
    region       VARCHAR NOT NULL,
    asset_class  VARCHAR NOT NULL,
    interval     VARCHAR NOT NULL,
    symbol       VARCHAR NOT NULL,
    first_ts     TIMESTAMPTZ,
    last_ts      TIMESTAMPTZ,
    row_count    BIGINT   NOT NULL DEFAULT 0,
    status       VARCHAR  NOT NULL DEFAULT 'ok',
    message      VARCHAR,
    updated_at   TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (provider, region, asset_class, interval, symbol)
);
"""

STATE_COLUMNS = (
    "provider",
    "region",
    "asset_class",
    "interval",
    "symbol",
    "first_ts",
    "last_ts",
    "row_count",
    "status",
    "message",
    "updated_at",
)


@dataclass
class SymbolState:
    symbol: str
    last_ts: pd.Timestamp | None
    row_count: int


@contextmanager
def connect(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(CATALOG_PATH), read_only=read_only)
    try:
        if not read_only:
            connection.execute(_STATE_DDL)
        ensure_views(connection)
        yield connection
    finally:
        connection.close()


def ensure_views(connection: duckdb.DuckDBPyConnection) -> None:
    """(Re)create the ``bars`` and ``universe`` views if their files exist."""
    if any(BARS_ROOT.glob("region=*/asset_class=*/interval=*/*.parquet")):
        pattern = (BARS_ROOT / "**" / "*.parquet").as_posix()
        connection.execute(
            f"""
            CREATE OR REPLACE VIEW bars AS
            SELECT * FROM read_parquet('{pattern}', hive_partitioning = true, union_by_name = true)
            """
        )
    universe_path = REFERENCE_ROOT / "universe.parquet"
    if universe_path.exists():
        connection.execute(
            f"CREATE OR REPLACE VIEW universe AS "
            f"SELECT * FROM read_parquet('{universe_path.as_posix()}')"
        )


def query(sql: str, parameters: Sequence[object] | None = None) -> pd.DataFrame:
    with connect(read_only=False) as connection:
        return connection.execute(sql, parameters or []).fetch_df()


def watermarks(provider: str, partition: Partition) -> dict[str, SymbolState]:
    """Newest stored timestamp per symbol, used to resume incremental downloads."""
    with connect() as connection:
        frame = connection.execute(
            """
            SELECT symbol, last_ts, row_count
            FROM ingest_state
            WHERE provider = ? AND region = ? AND asset_class = ? AND interval = ?
              AND last_ts IS NOT NULL
            """,
            [provider, partition.region, partition.asset_class, partition.interval],
        ).fetch_df()

    return {
        row.symbol: SymbolState(
            symbol=row.symbol,
            last_ts=pd.Timestamp(row.last_ts).tz_convert("UTC")
            if pd.notna(row.last_ts)
            else None,
            row_count=int(row.row_count),
        )
        for row in frame.itertuples()
    }


def record_states(rows: Iterable[dict[str, object]]) -> int:
    """Upsert ingest results; ``rows`` uses the :data:`STATE_COLUMNS` keys."""
    frame = pd.DataFrame(list(rows), columns=list(STATE_COLUMNS))
    if frame.empty:
        return 0
    for column in ("first_ts", "last_ts", "updated_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)

    with connect() as connection:
        connection.register("incoming_state", frame)
        # A failed refresh must not erase a good watermark, so coverage columns coalesce.
        connection.execute(
            """
            INSERT OR REPLACE INTO ingest_state
            SELECT i.provider, i.region, i.asset_class, i.interval, i.symbol,
                   coalesce(i.first_ts, s.first_ts),
                   coalesce(i.last_ts, s.last_ts),
                   CASE WHEN i.last_ts IS NULL THEN coalesce(s.row_count, 0) ELSE i.row_count END,
                   i.status, i.message, i.updated_at
            FROM incoming_state AS i
            LEFT JOIN ingest_state AS s
              ON  s.provider = i.provider AND s.region = i.region
              AND s.asset_class = i.asset_class AND s.interval = i.interval
              AND s.symbol = i.symbol
            """
        )
        connection.unregister("incoming_state")
    return len(frame)


def refresh_from_lake(provider: str = "yahoo") -> int:
    """Rebuild ``ingest_state`` by scanning the parquet files themselves."""
    now = pd.Timestamp.now(tz="UTC")
    rows: list[dict[str, object]] = []
    for partition in lake.iter_partitions():
        pattern = (partition.path / "*.parquet").as_posix()
        with connect() as connection:
            summary = connection.execute(
                f"""
                SELECT symbol, min(ts) AS first_ts, max(ts) AS last_ts, count(*) AS row_count
                FROM read_parquet('{pattern}')
                GROUP BY symbol
                """
            ).fetch_df()
        for row in summary.itertuples():
            rows.append(
                {
                    "provider": provider,
                    "region": partition.region,
                    "asset_class": partition.asset_class,
                    "interval": partition.interval,
                    "symbol": row.symbol,
                    "first_ts": row.first_ts,
                    "last_ts": row.last_ts,
                    "row_count": int(row.row_count),
                    "status": "ok",
                    "message": None,
                    "updated_at": now,
                }
            )
    return record_states(rows)


def status(interval: str | None = None) -> pd.DataFrame:
    sql = """
        SELECT provider, region, asset_class, interval,
               count(*) AS symbols,
               sum(row_count) AS rows,
               min(first_ts) AS first_ts,
               max(last_ts) AS last_ts,
               count(*) FILTER (WHERE status <> 'ok') AS failures
        FROM ingest_state
    """
    parameters: list[object] = []
    if interval:
        sql += " WHERE interval = ?"
        parameters.append(interval)
    sql += " GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4"
    return query(sql, parameters)


def export_parquet(sql: str, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect() as connection:
        connection.execute(
            f"COPY ({sql}) TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    return path
