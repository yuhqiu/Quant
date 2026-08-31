"""Metrics against the real lake. Skipped unless ``QUANT_LIVE_TESTS=1``."""

from __future__ import annotations

import os

import numpy as np
import pytest

from Common.io import matrix_path, read_matrix
from Common.types import Partition
from DataAcquisition import stored_symbols
from MetricsGeneration import build, min_periods, read_manifest
from MetricsGeneration.indicators import compute_metrics
from MetricsGeneration.storage import load_bars

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("QUANT_LIVE_TESTS") != "1",
        reason="set QUANT_LIVE_TESTS=1 to run tests against the real lake",
    ),
]

PARTITION = Partition("US", "stock", "1d")


@pytest.fixture(scope="module")
def manifest() -> dict:
    payload = read_manifest(PARTITION.metrics_dir)
    if payload is None:
        pytest.skip("no feature panel built; run 'python -m MetricsGeneration build'")
    return payload


class TestPanel:
    def test_manifest_describes_the_build(self, manifest):
        assert manifest["symbols"] > 0
        assert manifest["rows"] > 0
        assert manifest["partition"] == PARTITION.key
        assert manifest["start"] < manifest["end"]

    def test_manifest_carries_provenance(self, manifest):
        assert "versions" in manifest
        assert "created_utc" in manifest

    def test_every_declared_metric_is_on_disk(self, manifest):
        for metric in manifest["metrics"]:
            assert matrix_path(PARTITION.metrics_dir, metric).exists(), metric

    def test_labels_are_in_their_own_directory(self, manifest):
        for horizon in (1, 5, 21):
            assert matrix_path(PARTITION.labels_dir, f"fwd_ret_{horizon}d").exists()
            assert not matrix_path(PARTITION.metrics_dir, f"fwd_ret_{horizon}d").exists()

    def test_matrices_share_the_date_index(self, manifest):
        close = read_matrix(matrix_path(PARTITION.metrics_dir, "close"))
        rsi = read_matrix(matrix_path(PARTITION.metrics_dir, "rsi_14"))
        assert close.index.equals(rsi.index)
        assert list(close.columns) == list(rsi.columns)

    def test_index_is_utc_and_sorted(self, manifest):
        close = read_matrix(matrix_path(PARTITION.metrics_dir, "close"))
        assert str(close.index.tz) == "UTC"
        assert close.index.is_monotonic_increasing

    def test_adjusted_close_differs_from_raw_somewhere(self, manifest):
        close = read_matrix(matrix_path(PARTITION.metrics_dir, "close"))
        adjusted = read_matrix(matrix_path(PARTITION.metrics_dir, "adj_close"))
        assert (close.to_numpy() != adjusted.to_numpy()).any()

    def test_cross_sectional_ranks_are_bounded(self, manifest):
        ranks = read_matrix(matrix_path(PARTITION.metrics_dir, "cs_rank_mom_12_1"))
        values = ranks.to_numpy()
        finite = values[np.isfinite(values)]
        assert finite.min() >= 0.0
        assert finite.max() <= 1.0


class TestAgainstTheLake:
    def test_a_symbol_matches_a_direct_recomputation(self, manifest):
        symbol = next(s for s in manifest["symbol_list"] if s in {"AAPL", "MSFT", "IBM"})
        bars = load_bars(symbol, PARTITION)
        expected = compute_metrics(bars)["rsi_14"].dropna()

        stored = read_matrix(matrix_path(PARTITION.metrics_dir, "rsi_14"), columns=[symbol])
        actual = stored[symbol].reindex(expected.index)
        assert np.allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-4, equal_nan=True)

    def test_warm_up_is_respected(self, manifest):
        symbol = manifest["symbol_list"][0]
        bars = load_bars(symbol, PARTITION)
        if len(bars) < 300:
            pytest.skip("symbol is too short to test warm-up")
        metrics = compute_metrics(bars)
        for name in ("vol_252d", "mom_12_1", "sharpe_252d"):
            assert metrics[name].iloc[: min_periods(name) - 1].isna().all()


class TestIncremental:
    def test_second_build_is_skipped_when_nothing_is_new(self, manifest, tmp_path):
        symbols = stored_symbols(PARTITION)[:5]
        out = tmp_path / "metrics"
        labels = tmp_path / "labels"

        build(PARTITION, out, labels, symbols=symbols, workers=2, verbose=False)
        first = read_manifest(out)
        build(PARTITION, out, labels, symbols=symbols, workers=2, incremental=True, verbose=False)
        second = read_manifest(out)
        assert first["created_utc"] == second["created_utc"]

    def test_building_twice_produces_identical_files(self, manifest, tmp_path):
        from Common.provenance import file_hash

        symbols = stored_symbols(PARTITION)[:5]
        first = tmp_path / "a"
        second = tmp_path / "b"
        build(PARTITION, first, tmp_path / "la", symbols=symbols, workers=2, verbose=False)
        build(PARTITION, second, tmp_path / "lb", symbols=symbols, workers=2, verbose=False)

        for path in sorted(first.glob("*.parquet")):
            assert file_hash(path) == file_hash(second / path.name), path.name
