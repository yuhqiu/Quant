"""DataAcquisition offline regression: no network, one deterministic fake provider.

The ingest pipeline is inherently sequential -- a watermark only exists after a
download -- so the whole flow runs once in a module-scoped fixture and the tests
assert against what it observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Common.types import Partition
from DataAcquisition import catalog, lake, quality
from DataAcquisition.cleaning import clean_bars, merge_bars
from DataAcquisition.ingest import ingest
from DataAcquisition.lake import symbol_filename
from DataAcquisition.providers import get_provider, provider_names, register_provider
from DataAcquisition.providers.base import FetchRequest, FetchResult, MarketDataProvider
from DataAcquisition.schema import BAR_COLUMNS, normalize_bars
from DataAcquisition.test.harness import isolated_data_root

PARTITION = Partition("US", "stock", "1d")


def vendor_bars(start: str, periods: int, dividend: float = 0.0) -> pd.DataFrame:
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
            frame = vendor_bars("2024-01-01", FakeProvider.periods, dividend=0.25)
            if symbol == "DIRTY":
                frame.iloc[3, frame.columns.get_loc("High")] = 0.5   # high below low
                frame.iloc[5, frame.columns.get_loc("Close")] = -1.0
            if start is not None:
                frame = frame[frame.index >= start]
            if frame.empty:
                result.errors[symbol] = "no data returned"
            else:
                result.frames[symbol] = frame
        return result


register_provider(FakeProvider)


@pytest.fixture(scope="module")
def temporary_lake():
    with isolated_data_root("da_offline_") as root:
        yield root


@pytest.fixture(scope="module")
def pipeline(temporary_lake) -> dict:
    """Run the full download / resume / recover / rebuild flow once."""
    FakeProvider.periods = 40
    FakeProvider.fail_all = False
    FakeProvider.calls.clear()

    observed: dict[str, object] = {}
    observed["full"] = ingest(
        symbols=["AAPL", "CON", "BAD", "DIRTY"], provider="fake", mode="full", start="2024-01-01"
    )

    FakeProvider.calls.clear()
    observed["resume"] = ingest(symbols=["AAPL", "CON", "BAD"], provider="fake", mode="incremental")
    observed["resume_calls"] = {
        symbol: start for symbols, start in FakeProvider.calls for symbol in symbols
    }
    observed["watermark"] = lake.last_timestamp("AAPL", PARTITION)

    FakeProvider.periods = 50
    observed["append"] = ingest(symbols=["AAPL"], provider="fake", mode="incremental")
    observed["repeat"] = ingest(symbols=["AAPL"], provider="fake", mode="incremental")

    observed["before_outage"] = catalog.watermarks("fake", PARTITION)["AAPL"].last_ts
    FakeProvider.fail_all = True
    ingest(symbols=["AAPL"], provider="fake", mode="incremental")
    FakeProvider.fail_all = False
    observed["after_outage"] = catalog.watermarks("fake", PARTITION).get("AAPL")
    observed["recovered"] = ingest(symbols=["AAPL"], provider="fake", mode="incremental")
    return observed


class TestRegistry:
    def test_providers_are_registered(self):
        assert {"fake", "yahoo"} <= set(provider_names())

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValueError):
            get_provider("bloomberg")

    def test_unsupported_interval_is_rejected(self):
        with pytest.raises(ValueError):
            get_provider("fake").validate_interval("5m")

    def test_a_single_bad_symbol_does_not_raise(self):
        result = FakeProvider().fetch(FetchRequest(symbols=("AAPL", "BAD"), interval="1d"))
        assert "AAPL" in result.frames
        assert "BAD" in result.errors


class TestSchema:
    def test_normalize_produces_canonical_columns(self):
        frame = normalize_bars(vendor_bars("2024-01-01", 5), "AAPL")
        assert list(frame.columns) == list(BAR_COLUMNS)
        assert str(frame["ts"].dt.tz) == "UTC"

    def test_windows_reserved_names_are_escaped(self):
        assert symbol_filename("CON") == "CON_.parquet"
        assert symbol_filename("AAPL") == "AAPL.parquet"


class TestCleaning:
    def test_invalid_rows_are_dropped(self):
        frame = normalize_bars(vendor_bars("2024-01-01", 10), "X")
        frame.loc[2, "high"] = 0.5
        frame.loc[4, "close"] = -1.0
        assert len(clean_bars(frame)) == 8

    def test_merge_prefers_the_newer_bar(self):
        old = normalize_bars(vendor_bars("2024-01-01", 5), "X")
        new = old.copy()
        new["close"] = 999.0
        merged = merge_bars(old, new)
        assert len(merged) == 5
        assert (merged["close"] == 999.0).all()


class TestFullDownload:
    def test_clean_and_dirty_symbols_are_written(self, pipeline):
        assert pipeline["full"].updated == 3

    def test_the_bad_symbol_is_reported_not_raised(self, pipeline):
        assert "BAD" in pipeline["full"].failures

    def test_reserved_filename_lands_on_disk(self, pipeline):
        assert (PARTITION.path / "CON_.parquet").exists()

    def test_stored_symbols_round_trip(self, pipeline):
        assert lake.stored_symbols(PARTITION) == ["AAPL", "CON", "DIRTY"]

    def test_stored_frame_matches_the_canonical_schema(self, pipeline):
        frame = lake.read_symbol("AAPL", PARTITION)
        assert list(frame.columns) == list(BAR_COLUMNS)
        assert str(frame["ts"].dt.tz) == "UTC"

    def test_adjustments_are_stored_alongside_raw_prices(self, pipeline):
        frame = lake.read_symbol("AAPL", PARTITION)
        assert np.allclose(frame["adj_factor"], 0.99)
        assert frame["dividend"].gt(0).all()
        assert not np.allclose(frame["close"], frame["adj_close"])

    def test_dirty_rows_are_cleaned_before_write(self, pipeline):
        dirty = lake.read_symbol("DIRTY", PARTITION)
        assert len(dirty) == 38
        assert dirty[["open", "high", "low", "close"]].gt(0).all().all()


class TestIncremental:
    def test_resumes_from_the_watermark(self, pipeline):
        assert pipeline["resume_calls"]["AAPL"] == pipeline["watermark"].date().isoformat()

    def test_unchanged_source_adds_no_rows(self, pipeline):
        assert pipeline["resume"].rows_added == 0
        assert pipeline["resume"].unchanged == 2

    def test_only_new_bars_are_appended(self, pipeline):
        assert pipeline["append"].rows_added == 10
        frame = lake.read_symbol("AAPL", PARTITION)
        assert len(frame) == 50
        assert not frame["ts"].duplicated().any()
        assert frame["ts"].is_monotonic_increasing

    def test_ingesting_twice_adds_zero_rows(self, pipeline):
        assert pipeline["repeat"].rows_added == 0
        assert pipeline["repeat"].unchanged == 1


class TestDurability:
    def test_a_failed_run_keeps_the_watermark(self, pipeline):
        assert pipeline["after_outage"] is not None
        assert pipeline["after_outage"].last_ts == pipeline["before_outage"]

    def test_the_next_run_recovers(self, pipeline):
        assert not pipeline["recovered"].failures


class TestCatalog:
    def test_status_reports_the_partition(self, pipeline):
        status = catalog.status()
        assert len(status) == 1
        assert int(status.loc[0, "symbols"]) == 4

    def test_hive_partitions_are_queryable(self, pipeline):
        sql = catalog.query(
            "SELECT region, interval, symbol, count(*) AS n FROM bars GROUP BY ALL ORDER BY symbol"
        )
        assert set(sql["region"]) == {"US"}
        assert set(sql["interval"]) == {"1d"}
        assert sorted(sql["symbol"]) == ["AAPL", "CON", "DIRTY"]

    def test_catalog_is_rebuildable_from_the_lake(self, pipeline, temporary_lake):
        watermark = catalog.watermarks("fake", PARTITION)["AAPL"].last_ts
        (temporary_lake / "lake" / "catalog.duckdb").unlink(missing_ok=True)
        assert catalog.refresh_from_lake(provider="fake") == 3
        assert catalog.watermarks("fake", PARTITION)["AAPL"].last_ts == watermark


class TestQuality:
    def test_one_row_per_symbol(self, pipeline):
        assert len(quality.build_report(PARTITION)) == 3

    def test_expected_columns_are_present(self, pipeline):
        report = quality.build_report(PARTITION)
        assert {"stale_days", "max_gap_days", "price_jumps", "repaired_ratio"} <= set(report.columns)

    def test_report_is_persisted(self, pipeline):
        saved = quality.save_report(quality.build_report(PARTITION), PARTITION)
        assert saved.exists()
