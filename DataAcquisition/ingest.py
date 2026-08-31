"""Ingest orchestration: resolve symbols, fetch, clean, merge and record state.

``mode='incremental'`` asks the catalog for the newest stored bar per symbol and
only requests data from that point on, re-fetching the last stored bar so a
partial final candle is corrected rather than duplicated.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import pandas as pd

from . import catalog, lake, universe as universe_module
from .cleaning import clean_bars, merge_bars
from .config import (
    DEFAULT_ASSET_CLASS,
    DEFAULT_INTERVAL,
    DEFAULT_PROVIDER,
    DEFAULT_REGION,
)
from .lake import Partition
from .providers import FetchRequest, MarketDataProvider, get_provider
from .schema import TIMESTAMP, normalize_bars

Mode = Literal["auto", "full", "incremental"]

ProgressCallback = Callable[[int, int, str, str], None]


@dataclass
class IngestReport:
    partition: Partition
    provider: str
    requested: int = 0
    updated: int = 0
    unchanged: int = 0
    rows_added: int = 0
    failures: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.provider} {self.partition.region}/{self.partition.asset_class}/"
            f"{self.partition.interval}: {self.updated} updated, {self.unchanged} unchanged, "
            f"{len(self.failures)} failed, +{self.rows_added} rows"
        )


def resolve_symbols(
    symbols: str | Iterable[str] | None = None,
    symbols_file: str | Path | None = None,
    asset_class: str | None = None,
    region: str = DEFAULT_REGION,
    refresh_universe: bool = False,
) -> list[str]:
    """Combine explicit symbols, a symbols file and a universe into one unique list."""
    collected: list[str] = []
    if asset_class:
        collected.extend(
            universe_module.symbols(asset_class, region=region, refresh=refresh_universe)
        )
    if symbols_file:
        text = Path(symbols_file).read_text(encoding="utf-8")
        collected.extend(part for part in text.replace(",", " ").split())
    if symbols:
        collected.extend([symbols] if isinstance(symbols, str) else symbols)

    unique: dict[str, None] = {}
    for symbol in collected:
        ticker = str(symbol).strip().upper()
        if ticker:
            unique.setdefault(ticker, None)
    if not unique:
        raise ValueError("provide symbols, symbols_file or asset_class")
    return list(unique)


def _plan_starts(
    tickers: Sequence[str],
    partition: Partition,
    provider_name: str,
    mode: Mode,
    start: pd.Timestamp | None,
) -> dict[str, pd.Timestamp | None]:
    """Map each symbol to the timestamp its download should begin at."""
    if mode == "full":
        return {ticker: start for ticker in tickers}

    states = catalog.watermarks(provider_name, partition)
    if mode == "incremental" and not states:
        raise ValueError(
            "incremental mode found no catalog state; run a 'full' download first "
            "or use mode='auto'"
        )

    planned: dict[str, pd.Timestamp | None] = {}
    for ticker in tickers:
        state = states.get(ticker)
        if state is None or state.last_ts is None:
            planned[ticker] = start
        else:
            # Re-request the last stored bar so a partial candle gets overwritten.
            planned[ticker] = state.last_ts
    return planned


def _batches(items: Sequence[str], size: int) -> list[list[str]]:
    size = max(int(size), 1)
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def ingest(
    symbols: str | Iterable[str] | None = None,
    *,
    provider: str | MarketDataProvider = DEFAULT_PROVIDER,
    region: str = DEFAULT_REGION,
    asset_class: str = DEFAULT_ASSET_CLASS,
    interval: str = DEFAULT_INTERVAL,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    mode: Mode = "auto",
    symbols_file: str | Path | None = None,
    use_universe: bool = False,
    refresh_universe: bool = False,
    batch_size: int | None = None,
    pause: float | None = None,
    drop_zero_volume: bool = False,
    on_progress: ProgressCallback | None = None,
) -> IngestReport:
    """Download bars into the lake and update the catalog watermarks."""
    engine = get_provider(provider) if isinstance(provider, str) else provider
    engine.validate_interval(interval)

    tickers = resolve_symbols(
        symbols=symbols,
        symbols_file=symbols_file,
        asset_class=asset_class if use_universe else None,
        region=region,
        refresh_universe=refresh_universe,
    )

    partition = Partition(region=region, asset_class=asset_class, interval=interval)
    report = IngestReport(partition=partition, provider=engine.name, requested=len(tickers))

    requested_start = pd.Timestamp(start, tz="UTC") if start is not None else None
    requested_end = pd.Timestamp(end, tz="UTC") if end is not None else None
    planned = _plan_starts(tickers, partition, engine.name, mode, requested_start)

    # Symbols sharing a start date can travel in the same provider request.
    grouped: dict[pd.Timestamp | None, list[str]] = defaultdict(list)
    for ticker, ticker_start in planned.items():
        clamped = engine.clamp_start(interval, ticker_start)
        key = clamped.normalize() if clamped is not None else None
        grouped[key].append(ticker)

    size = batch_size if batch_size is not None else engine.max_batch_size
    delay = pause if pause is not None else engine.request_pause
    now = pd.Timestamp.now(tz="UTC")
    state_rows: list[dict[str, object]] = []
    done = 0

    requests = [
        (group_start, batch)
        for group_start, group in sorted(grouped.items(), key=lambda item: (item[0] is not None, item[0]))
        for batch in _batches(group, size)
    ]

    for index, (group_start, batch) in enumerate(requests):
        result = engine.fetch(
            FetchRequest(
                symbols=tuple(batch),
                interval=interval,
                start=group_start,
                end=requested_end,
            )
        )

        for ticker in batch:
            done += 1
            error = result.errors.get(ticker)
            if error is not None:
                report.failures[ticker] = error
                state_rows.append(
                    _state_row(engine.name, partition, ticker, None, now, "error", error)
                )
                if on_progress:
                    on_progress(done, len(tickers), ticker, f"FAILED - {error}")
                continue

            try:
                incoming = clean_bars(
                    normalize_bars(result.frames[ticker], ticker),
                    drop_zero_volume=drop_zero_volume,
                )
                if incoming.empty:
                    raise ValueError("no rows left after cleaning")

                existing = lake.read_symbol(ticker, partition)
                before = len(existing)
                merged = merge_bars(existing, incoming)
                added = len(merged) - before

                if added == 0 and before and merged[TIMESTAMP].max() <= existing[TIMESTAMP].max():
                    report.unchanged += 1
                else:
                    report.updated += 1
                    report.rows_added += max(added, 0)

                lake.write_symbol(merged, ticker, partition)
                state_rows.append(
                    _state_row(engine.name, partition, ticker, merged, now, "ok", None)
                )
                if on_progress:
                    on_progress(done, len(tickers), ticker, f"{len(merged)} rows (+{max(added, 0)})")
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                report.failures[ticker] = message
                state_rows.append(
                    _state_row(engine.name, partition, ticker, None, now, "error", message)
                )
                if on_progress:
                    on_progress(done, len(tickers), ticker, f"FAILED - {message}")

        if delay and index + 1 < len(requests):
            time.sleep(delay)

    catalog.record_states(state_rows)
    return report


def _state_row(
    provider: str,
    partition: Partition,
    symbol: str,
    frame: pd.DataFrame | None,
    now: pd.Timestamp,
    status: str,
    message: str | None,
) -> dict[str, object]:
    first_ts = last_ts = None
    row_count = 0
    if frame is not None and not frame.empty:
        first_ts = frame[TIMESTAMP].min()
        last_ts = frame[TIMESTAMP].max()
        row_count = len(frame)
    return {
        "provider": provider,
        "region": partition.region,
        "asset_class": partition.asset_class,
        "interval": partition.interval,
        "symbol": symbol,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "row_count": row_count,
        "status": status,
        "message": message,
        "updated_at": now,
    }
