"""Integration tests against real vendors and the real lake.

Skipped unless ``QUANT_LIVE_TESTS=1``, because they hit the network and depend on
data that changes underneath them.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("QUANT_LIVE_TESTS") != "1",
        reason="set QUANT_LIVE_TESTS=1 to run live vendor tests",
    ),
]

from Common.types import Partition  # noqa: E402
from DataAcquisition import catalog, lake, quality, universe  # noqa: E402
from DataAcquisition.providers import get_provider  # noqa: E402
from DataAcquisition.providers.base import FetchRequest  # noqa: E402
from DataAcquisition.schema import BAR_COLUMNS  # noqa: E402

PARTITION = Partition("US", "stock", "1d")


class TestVendor:
    def test_yahoo_returns_usable_daily_bars(self):
        provider = get_provider("yahoo")
        result = provider.fetch(
            FetchRequest(symbols=("AAPL", "MSFT"), interval="1d", start=pd.Timestamp("2024-01-01", tz="UTC"))
        )
        assert set(result.frames) == {"AAPL", "MSFT"}
        for frame in result.frames.values():
            assert len(frame) > 100

    def test_a_nonsense_ticker_is_an_error_not_an_exception(self):
        result = get_provider("yahoo").fetch(
            FetchRequest(symbols=("ZZZZNOTREAL",), interval="1d")
        )
        assert "ZZZZNOTREAL" in result.errors or not result.frames


class TestUniverse:
    def test_nasdaq_directory_downloads(self):
        listing = universe.build_universe()
        assert len(listing) > 5_000
        assert set(listing["asset_class"]) <= {"stock", "etf", "other"}
        assert "snapshot_date" in listing.columns


class TestRealLake:
    def test_lake_has_symbols(self):
        symbols = lake.stored_symbols(PARTITION)
        if not symbols:
            pytest.skip("no bars stored yet")
        assert len(symbols) > 0

    def test_stored_bars_match_the_schema(self):
        symbols = lake.stored_symbols(PARTITION)
        if not symbols:
            pytest.skip("no bars stored yet")
        frame = lake.read_symbol(symbols[0], PARTITION)
        assert list(frame.columns) == list(BAR_COLUMNS)
        assert frame["ts"].is_monotonic_increasing

    def test_catalog_status_is_readable(self):
        status = catalog.status()
        assert isinstance(status, pd.DataFrame)

    def test_quality_report_builds(self):
        symbols = lake.stored_symbols(PARTITION)
        if not symbols:
            pytest.skip("no bars stored yet")
        report = quality.build_report(PARTITION)
        assert len(report) > 0
        assert report["rows"].min() >= 0
