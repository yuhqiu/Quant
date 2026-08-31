"""Indicators checked against hand-computed fixtures, not against themselves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from MetricsGeneration.indicators import (
    adjusted_ohlc,
    compute_labels,
    compute_metrics,
    metric_dtype,
    rsi,
    true_range,
    with_adjustments,
)


def frame_from(closes: list[float], **columns) -> pd.DataFrame:
    index = pd.bdate_range("2021-01-04", periods=len(closes), tz="UTC", name="date")
    data = {
        "open": columns.get("open", closes),
        "high": columns.get("high", closes),
        "low": columns.get("low", closes),
        "close": closes,
        "volume": columns.get("volume", [1_000_000.0] * len(closes)),
    }
    frame = pd.DataFrame(data, index=index, dtype="float64")
    for name, values in columns.items():
        if name not in frame.columns:
            frame[name] = values
    return frame


class TestGroundTruth:
    def test_true_range_by_hand(self):
        high = pd.Series([10.0, 12.0, 11.0])
        low = pd.Series([9.0, 9.5, 10.0])
        close = pd.Series([9.5, 11.0, 10.5])
        # bar 0: high-low = 1. bar 1: max(2.5, |12-9.5|, |9.5-9.5|) = 2.5.
        # bar 2: max(1.0, |11-11|, |10-11|) = 1.0.
        assert true_range(high, low, close).tolist() == pytest.approx([1.0, 2.5, 1.0])

    def test_rsi_of_a_pure_uptrend_is_100(self):
        rising = pd.Series(np.arange(100.0, 130.0))
        assert rsi(rising, 14).iloc[-1] == pytest.approx(100.0)

    def test_rsi_of_a_pure_downtrend_is_zero(self):
        falling = pd.Series(np.arange(130.0, 100.0, -1.0))
        assert rsi(falling, 14).iloc[-1] == pytest.approx(0.0)

    def test_simple_return_is_the_price_ratio(self):
        frame = frame_from([100.0, 110.0, 99.0])
        metrics = compute_metrics(frame)
        assert metrics["ret_1d"].tolist()[1:] == pytest.approx([0.10, -0.10])

    def test_sma_distance_by_hand(self):
        prices = [10.0] * 9 + [20.0]
        metrics = compute_metrics(frame_from(prices))
        # 10-day mean is (9*10 + 20)/10 = 11; 20/11 - 1 = 0.818181...
        assert metrics["px_to_sma_10"].iloc[-1] == pytest.approx(20.0 / 11.0 - 1.0)

    def test_dollar_volume_uses_the_raw_traded_price(self):
        frame = frame_from([50.0, 50.0], volume=[1000.0, 2000.0])
        frame["adj_close"] = [25.0, 25.0]
        metrics = compute_metrics(frame)
        assert metrics["dollar_vol"].tolist() == pytest.approx([50_000.0, 100_000.0])

    def test_bollinger_percent_b_by_hand(self):
        prices = [100.0, 101.0] * 10
        metrics = compute_metrics(frame_from(prices))
        # 10 values at each level: mean 100.5, sample std sqrt(5/19).
        deviation = np.sqrt(5.0 / 19.0)
        expected = (101.0 - (100.5 - 2.0 * deviation)) / (4.0 * deviation)
        assert metrics["bb_pctb_20"].iloc[-1] == pytest.approx(expected)
        assert metrics["zscore_20"].iloc[-1] == pytest.approx(0.5 / deviation)


class TestAdjustments:
    def test_returns_ignore_a_two_for_one_split(self):
        # Raw price halves; the adjusted series does not, so the return is zero.
        frame = frame_from([100.0, 50.0])
        frame["adj_close"] = [50.0, 50.0]
        frame["split_ratio"] = [0.0, 2.0]
        metrics = compute_metrics(frame)
        assert metrics["ret_1d"].iloc[1] == pytest.approx(0.0)

    def test_adjusted_ohlc_scales_by_the_same_factor(self):
        frame = with_adjustments(frame_from([100.0, 50.0], open=[100.0, 50.0]))
        frame["adj_close"] = [50.0, 50.0]
        frame["adj_factor"] = frame["adj_close"] / frame["close"]
        adjusted = adjusted_ohlc(frame)
        assert adjusted["open"].tolist() == pytest.approx([50.0, 50.0])
        assert adjusted["close"].tolist() == pytest.approx([50.0, 50.0])

    def test_missing_adjustment_columns_fall_back_to_raw(self):
        frame = with_adjustments(frame_from([10.0, 11.0]))
        assert frame["adj_close"].tolist() == [10.0, 11.0]
        assert frame["adj_factor"].tolist() == pytest.approx([1.0, 1.0])

    def test_labels_use_the_adjusted_series(self):
        frame = frame_from([100.0, 50.0, 50.0])
        frame["adj_close"] = [50.0, 50.0, 50.0]
        labels = compute_labels(frame)
        assert labels["fwd_ret_1d"].iloc[0] == pytest.approx(0.0)


class TestDtypes:
    def test_prices_keep_double_precision(self):
        assert metric_dtype("close") == "float64"
        assert metric_dtype("adj_close") == "float64"
        assert metric_dtype("dividend") == "float64"

    def test_derived_features_are_single_precision(self):
        assert metric_dtype("rsi_14") == "float32"

    def test_no_infinities_survive(self, bars):
        metrics = compute_metrics(bars)
        assert np.isfinite(metrics.to_numpy(dtype="float64")[~metrics.isna().to_numpy()]).all()


class TestIdempotence:
    def test_computing_twice_gives_identical_output(self, bars):
        pd.testing.assert_frame_equal(compute_metrics(bars), compute_metrics(bars))

    def test_leading_values_stay_nan(self, bars):
        metrics = compute_metrics(bars)
        assert metrics["vol_252d"].iloc[:250].isna().all()
        assert metrics["mom_12_1"].iloc[:250].isna().all()
