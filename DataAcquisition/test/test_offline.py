"""Offline regression test for DataAcquisition: no network, deterministic provider.

    python DataAcquisition\\test\\test_offline.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import PROJECT_ROOT, Results, isolated_data_root  # noqa: E402

LAKE = isolated_data_root("da_offline_")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from DataAcquisition import catalog, lake, quality  # noqa: E402
from DataAcquisition.ingest import ingest  # noqa: E402
from DataAcquisition.lake import Partition  # noqa: E402
from DataAcquisition.migrate import migrate_csv_directory  # noqa: E402
from DataAcquisition.providers import get_provider, provider_names, register_provider  # noqa: E402
from DataAcquisition.providers.base import FetchRequest, FetchResult, MarketDataProvider  # noqa: E402
from DataAcquisition.schema import BAR_COLUMNS  # noqa: E402

results = Results()
check = results.check
PART = Partition("US", "stock", "1d")


def bars(start: str, periods: int, dividend: float = 0.0) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=periods, tz="UTC")
    base = np.linspace(10.0, 20.0, periods)
    return pd.DataFrame(
        {
            "Open": base,
            "High": base * 1.02,
            "Low": base * 0.98,
            "Close": base * 1.01,
            "Adj Close": base * 1.01 * 0.99,
            "Volume": np.full(periods, 1_000.0),
            "Dividends": np.full(periods, dividend),
            "Stock Splits": 0.0,
            "Repaired?": False,
        },
        index=index,
    )


class FakeProvider(MarketDataProvider):
    """Serves a fixed history slice, the way a real vendor responds to a start date."""

    name = "fake"
    intervals = frozenset({"1d"})
    max_batch_size = 2
    request_pause = 0.0
    fail_all = False
    periods = 40
    calls: list[tuple[tuple[str, ...], str | None]] = []

    def fetch(self, request: FetchRequest) -> FetchResult:
        start = request.start
        FakeProvider.calls.append(
            (request.symbols, start.date().isoformat() if start is not None else None)
        )
        result = FetchResult()
        if FakeProvider.fail_all:
            result.errors = {symbol: "provider down" for symbol in request.symbols}
            return result

        for symbol in request.symbols:
            if symbol == "BAD":
                result.errors[symbol] = "no data returned"
                continue
            frame = bars("2024-01-01", FakeProvider.periods, dividend=0.25)
            if symbol == "DIRTY":
                frame.iloc[3, frame.columns.get_loc("High")] = 0.5  # high < low
                frame.iloc[5, frame.columns.get_loc("Close")] = -1.0
            if start is not None:
                frame = frame[frame.index >= start]
            if frame.empty:
                result.errors[symbol] = "no data returned"
            else:
                result.frames[symbol] = frame
        return result


register_provider(FakeProvider)


results.section("registry & interfaces")
check("fake provider registered", "fake" in provider_names(), provider_names())
check("yahoo provider registered", "yahoo" in provider_names())
try:
    get_provider("bloomberg")
    check("unknown provider rejected", False)
except ValueError as exc:
    check("unknown provider rejected", True, str(exc)[:60])
try:
    get_provider("fake").validate_interval("5m")
    check("unsupported interval rejected", False)
except ValueError:
    check("unsupported interval rejected", True)


results.section("full download")
run1 = ingest(
    symbols=["AAPL", "CON", "BAD", "DIRTY"], provider="fake", mode="full", start="2024-01-01"
)
print("  " + run1.summary())
check("clean and dirty symbols written", run1.updated == 3, f"updated={run1.updated}")
check("bad symbol reported", "BAD" in run1.failures, run1.failures.get("BAD"))
check("windows reserved name handled", (PART.path / "CON_.parquet").exists())
check("stored symbols round-trip", lake.stored_symbols(PART) == ["AAPL", "CON", "DIRTY"], lake.stored_symbols(PART))

frame = lake.read_symbol("AAPL", PART)
check("canonical columns", list(frame.columns) == list(BAR_COLUMNS))
check("timestamps are UTC", str(frame["ts"].dt.tz) == "UTC")
check("adj_factor derived", np.allclose(frame["adj_factor"], 0.99), f"{frame['adj_factor'].iloc[0]:.6f}")
check("dividends stored", frame["dividend"].gt(0).all(), f"sum={frame['dividend'].sum():.2f}")
check("raw close preserved", not np.allclose(frame["close"], frame["adj_close"]))

dirty = lake.read_symbol("DIRTY", PART)
check("invalid OHLC rows dropped", len(dirty) == FakeProvider.periods - 2, f"rows={len(dirty)}")
check("no non-positive prices", dirty[["open", "high", "low", "close"]].gt(0).all().all())


results.section("incremental download")
FakeProvider.calls.clear()
run2 = ingest(symbols=["AAPL", "CON", "BAD"], provider="fake", mode="incremental")
print("  " + run2.summary())
requested_starts = {symbol: start for symbols, start in FakeProvider.calls for symbol in symbols}
watermark = lake.last_timestamp("AAPL", PART)
check(
    "resumed from watermark, not from scratch",
    requested_starts["AAPL"] == watermark.date().isoformat(),
    requested_starts,
)
check("nothing new when source is unchanged", run2.rows_added == 0 and run2.unchanged == 2, run2.summary())

# The vendor publishes ten more sessions; only those should be fetched and appended.
FakeProvider.periods = 50
run3 = ingest(symbols=["AAPL"], provider="fake", mode="incremental")
appended = lake.read_symbol("AAPL", PART)
check("only new bars appended", run3.rows_added == 10, run3.summary())
check("total history correct", len(appended) == 50, f"rows={len(appended)}")
check("no duplicate timestamps", not appended["ts"].duplicated().any())
check("timestamps sorted", appended["ts"].is_monotonic_increasing)

run4 = ingest(symbols=["AAPL"], provider="fake", mode="incremental")
check("re-running is idempotent", run4.rows_added == 0 and run4.unchanged == 1, run4.summary())


results.section("watermark durability")
before = catalog.watermarks("fake", PART)["AAPL"].last_ts
FakeProvider.fail_all = True
ingest(symbols=["AAPL"], provider="fake", mode="incremental")
FakeProvider.fail_all = False
after = catalog.watermarks("fake", PART).get("AAPL")
check(
    "failed run keeps watermark",
    after is not None and after.last_ts == before,
    f"{before} -> {after and after.last_ts}",
)
recovered = ingest(symbols=["AAPL"], provider="fake", mode="incremental")
check("recovers after outage", not recovered.failures, recovered.summary())


results.section("catalog & sql")
status = catalog.status()
check(
    "status reports partition",
    len(status) == 1 and int(status.loc[0, "symbols"]) == 4,
    f"symbols={status.loc[0, 'symbols'] if len(status) else 0}",
)
sql = catalog.query(
    "SELECT region, asset_class, interval, symbol, count(*) AS n FROM bars GROUP BY ALL ORDER BY symbol"
)
check("hive partitions exposed in sql", set(sql["region"]) == {"US"} and set(sql["interval"]) == {"1d"})
check("all symbols queryable", sorted(sql["symbol"]) == ["AAPL", "CON", "DIRTY"], sorted(sql["symbol"]))

(LAKE / "lake" / "catalog.duckdb").unlink(missing_ok=True)
rebuilt = catalog.refresh_from_lake(provider="fake")
check("catalog rebuildable from lake", rebuilt == 3, f"entries={rebuilt}")
check("watermarks survive rebuild", catalog.watermarks("fake", PART)["AAPL"].last_ts == before)


results.section("quality report")
report = quality.build_report(PART)
check("one row per symbol", len(report) == 3, f"rows={len(report)}")
check(
    "expected report columns",
    {"stale_days", "max_gap_days", "price_jumps", "repaired_ratio"} <= set(report.columns),
)
saved = quality.save_report(report, PART)
check("report persisted as parquet", saved.exists(), saved.name)


results.section("csv migration")
legacy = LAKE / "legacy"
legacy.mkdir(parents=True, exist_ok=True)
for symbol in ("AAPL", "ABT", "ACN"):
    source = PROJECT_ROOT / "DataSource" / "US" / "Stock" / "day" / f"{symbol}.csv"
    if source.exists():
        shutil.copy(source, legacy / source.name)

if any(legacy.glob("*.csv")):
    target = Partition("US", "stock", "1d_legacy")
    counts = migrate_csv_directory(legacy, target)
    check("legacy csvs converted", counts["converted"] == 3 and counts["skipped"] == 0, counts)
    migrated = lake.read_symbol("AAPL", target)
    check("migrated rows carry schema", list(migrated.columns) == list(BAR_COLUMNS))
    check("migrated history intact", len(migrated) > 4000, f"rows={len(migrated)}")
else:
    print("  [SKIP] no legacy CSVs available")


code = results.exit_code()
shutil.rmtree(LAKE, ignore_errors=True)
sys.exit(code)
