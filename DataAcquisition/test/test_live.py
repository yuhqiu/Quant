"""Live network test for DataAcquisition: NASDAQ universe and Yahoo downloads.

    python DataAcquisition\\test\\test_live.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Results, isolated_data_root  # noqa: E402

LAKE = isolated_data_root("da_live_")

import pandas as pd  # noqa: E402

from DataAcquisition import catalog, lake, quality, universe  # noqa: E402
from DataAcquisition.ingest import ingest  # noqa: E402
from DataAcquisition.lake import Partition  # noqa: E402

results = Results()
check = results.check


results.section("universe (NASDAQ symbol directory)")
frame = universe.build_universe()
counts = frame["asset_class"].value_counts().to_dict()
universe.save_universe(frame)
check("stocks discovered", counts.get("stock", 0) > 3000, counts)
check("etfs discovered", counts.get("etf", 0) > 2000, f"etf={counts.get('etf')}")
check("symbols unique", not frame["symbol"].duplicated().any())
check("yahoo symbol format", not frame["symbol"].str.contains(r"\.").any())
stocks = universe.symbols("stock")
etfs = universe.symbols("etf")
check(
    "universe queryable by asset class",
    "AAPL" in stocks and "SPY" in etfs,
    f"{len(stocks)} stocks / {len(etfs)} etfs",
)


results.section("yahoo: daily stocks, explicit range")
stock_part = Partition("US", "stock", "1d")
run = ingest(
    symbols=["AAPL", "MSFT", "BRK-B", "NOT-A-REAL-TICKER"],
    provider="yahoo",
    asset_class="stock",
    interval="1d",
    start="2025-01-01",
    end="2025-07-01",
    mode="full",
)
print("  " + run.summary())
check("valid symbols downloaded", run.updated == 3, f"updated={run.updated}")
check("invalid symbol isolated", "NOT-A-REAL-TICKER" in run.failures)
aapl = lake.read_symbol("AAPL", stock_part)
check("respects start date", aapl["ts"].min() >= pd.Timestamp("2025-01-01", tz="UTC"), aapl["ts"].min())
check("respects end date", aapl["ts"].max() < pd.Timestamp("2025-07-01", tz="UTC"), aapl["ts"].max())
check("roughly a half year of sessions", 115 < len(aapl) < 130, f"rows={len(aapl)}")
check("dividends captured", aapl["dividend"].sum() > 0, f"sum={aapl['dividend'].sum():.3f}")
check("raw vs adjusted differ", not aapl["close"].equals(aapl["adj_close"]))
check("adj_factor in sane range", aapl["adj_factor"].between(0.5, 1.5).all())


results.section("yahoo: incremental top-up to today")
before = lake.last_timestamp("AAPL", stock_part)
update = ingest(symbols=["AAPL", "MSFT", "BRK-B"], provider="yahoo", asset_class="stock", interval="1d")
print("  " + update.summary())
after = lake.last_timestamp("AAPL", stock_part)
topped_up = lake.read_symbol("AAPL", stock_part)
check("watermark advanced", after > before, f"{before.date()} -> {after.date()}")
check("history extended, not replaced", len(topped_up) > len(aapl))
check("no duplicate timestamps", not topped_up["ts"].duplicated().any())
check("start of history preserved", topped_up["ts"].min() == aapl["ts"].min())

again = ingest(symbols=["AAPL"], provider="yahoo", asset_class="stock", interval="1d")
check("second update is a no-op", again.rows_added == 0, again.summary())


results.section("yahoo: etfs, weekly interval")
etf_part = Partition("US", "etf", "1wk")
etf_run = ingest(
    symbols=["SPY", "QQQ", "IWM"],
    provider="yahoo",
    asset_class="etf",
    interval="1wk",
    start="2024-01-01",
    mode="full",
)
print("  " + etf_run.summary())
spy = lake.read_symbol("SPY", etf_part)
check("etf partition written", etf_run.updated == 3 and not etf_run.failures)
check("weekly bars spaced ~7 days", spy["ts"].diff().dt.days.median() == 7, spy["ts"].diff().dt.days.median())
check("etf isolated from stock partition", "SPY" not in lake.stored_symbols(stock_part))


results.section("yahoo: intraday, lookback clamping")
intraday_part = Partition("US", "stock", "5m")
# Yahoo only serves ~60 days of 5m bars; the request start must be clamped, not rejected.
intraday = ingest(
    symbols=["AAPL"],
    provider="yahoo",
    asset_class="stock",
    interval="5m",
    start="2015-01-01",
    mode="full",
)
print("  " + intraday.summary())
if intraday.updated:
    bars_5m = lake.read_symbol("AAPL", intraday_part)
    span = (pd.Timestamp.now(tz="UTC") - bars_5m["ts"].min()).days
    check("intraday request clamped to provider limit", span <= 62, f"oldest bar {span} days back")
    check("intraday bars spaced 5 minutes", bars_5m["ts"].diff().dt.total_seconds().mode()[0] == 300)
else:
    check("intraday download", False, intraday.failures)


results.section("catalog & quality over live data")
status = catalog.status()
print(status.to_string(index=False))
check("three partitions tracked", len(status) == 3, f"partitions={len(status)}")
report = quality.build_report(stock_part)
check("quality report built", len(report) == 3, f"rows={len(report)}")
check("live data is fresh", int(report["stale_days"].max()) <= 5, f"max stale_days={report['stale_days'].max()}")
check("no price jumps flagged", int(report["price_jumps"].sum()) == 0, f"jumps={report['price_jumps'].sum()}")

cross = catalog.query(
    """
    SELECT asset_class, interval, count(DISTINCT symbol) AS symbols, count(*) AS bars
    FROM bars GROUP BY ALL ORDER BY asset_class, interval
    """
)
print(cross.to_string(index=False))
check("sql spans every partition", len(cross) == 3, f"groups={len(cross)}")


code = results.exit_code()
shutil.rmtree(LAKE, ignore_errors=True)
sys.exit(code)
