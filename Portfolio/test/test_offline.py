"""Portfolio: constructors, the constraint chain, and rebalance scheduling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Portfolio import build_targets
from Portfolio.base import MarketContext, normalize_gross
from Portfolio.constraints import (
    ConstraintChain,
    MaxBeta,
    MaxGross,
    MaxGroupWeight,
    MaxNet,
    MaxWeightPerName,
    MinPositions,
    build_chain,
)
from Portfolio.constructors import (
    InverseVolWeighted,
    MeanVariance,
    QuantileLongShort,
    ScoreProportional,
    TopNEqualWeight,
    get_constructor,
    ledoit_wolf,
)
from Portfolio.schedule import limit_turnover, no_trade_band, rebalance_dates

SYMBOLS = [f"S{i:02d}" for i in range(20)]


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=60, tz="UTC", name="date")


@pytest.fixture
def scores(dates) -> pd.DataFrame:
    values = np.tile(np.arange(len(SYMBOLS), dtype=float), (len(dates), 1))
    return pd.DataFrame(values, index=dates, columns=SYMBOLS)


@pytest.fixture
def context(dates) -> MarketContext:
    prices = pd.DataFrame(100.0, index=dates, columns=SYMBOLS)
    volatility = pd.DataFrame(
        np.tile(np.linspace(0.1, 0.5, len(SYMBOLS)), (len(dates), 1)),
        index=dates,
        columns=SYMBOLS,
    )
    beta = pd.DataFrame(1.0, index=dates, columns=SYMBOLS)
    returns = pd.DataFrame(
        np.random.default_rng(0).normal(0.0, 0.01, (len(dates), len(SYMBOLS))),
        index=dates,
        columns=SYMBOLS,
    )
    return MarketContext(prices=prices, volatility=volatility, beta=beta, returns=returns)


class TestConstructors:
    def test_top_n_holds_exactly_n_names(self, scores, context):
        weights = TopNEqualWeight(n=5).target_weights(scores, context)
        assert (weights > 0).sum(axis=1).eq(5).all()
        assert weights.sum(axis=1).eq(1.0).all()

    def test_top_n_picks_the_highest_scores(self, scores, context):
        weights = TopNEqualWeight(n=3).target_weights(scores, context)
        assert list(weights.iloc[0][weights.iloc[0] > 0].index) == SYMBOLS[-3:]

    def test_quantile_long_short_is_dollar_neutral(self, scores, context):
        weights = QuantileLongShort(quantiles=5, gross=1.0, net=0.0).target_weights(scores, context)
        assert weights.sum(axis=1).abs().max() == pytest.approx(0.0, abs=1e-12)
        assert weights.abs().sum(axis=1).max() == pytest.approx(1.0)

    def test_quantile_respects_a_net_tilt(self, scores, context):
        weights = QuantileLongShort(quantiles=5, gross=1.0, net=0.2).target_weights(scores, context)
        assert weights.sum(axis=1).max() == pytest.approx(0.2)
        assert weights.abs().sum(axis=1).max() == pytest.approx(1.0)

    def test_quantile_longs_the_top_and_shorts_the_bottom(self, scores, context):
        row = QuantileLongShort(quantiles=5).target_weights(scores, context).iloc[0]
        assert (row[SYMBOLS[-4:]] > 0).all()
        assert (row[SYMBOLS[:4]] < 0).all()

    def test_score_proportional_respects_the_cap(self, scores, context):
        weights = ScoreProportional(cap_per_name=0.06).target_weights(scores, context)
        assert weights.abs().max().max() <= 0.06 + 1e-12

    def test_inverse_vol_tilts_toward_calm_names(self, scores, context):
        weights = InverseVolWeighted(quantiles=5).target_weights(scores, context)
        longs = weights.iloc[0][weights.iloc[0] > 0]
        # Volatility rises with the symbol index, so the calmest long gets the most.
        assert longs.idxmin() == longs.index[-1]

    def test_mean_variance_produces_a_sized_book(self, scores, context):
        weights = MeanVariance(max_names=10, lookback=60).target_weights(scores.tail(5), context)
        assert weights.abs().sum(axis=1).max() <= 1.0 + 1e-9

    def test_tradability_mask_is_honoured(self, scores, context, dates):
        context.tradable = pd.DataFrame(True, index=dates, columns=SYMBOLS)
        context.tradable[SYMBOLS[-1]] = False
        weights = TopNEqualWeight(n=3).target_weights(scores, context)
        assert weights[SYMBOLS[-1]].abs().max() == 0.0

    def test_registry_lookup(self):
        assert isinstance(get_constructor("quantile_long_short", quantiles=3), QuantileLongShort)
        with pytest.raises(ValueError, match="unknown constructor"):
            get_constructor("nope")


class TestLedoitWolf:
    def test_shrinkage_is_positive_definite(self):
        sample = np.random.default_rng(0).normal(size=(80, 40))
        covariance = ledoit_wolf(sample)
        assert np.all(np.linalg.eigvalsh(covariance) > 0)

    def test_shrinkage_is_symmetric(self):
        covariance = ledoit_wolf(np.random.default_rng(1).normal(size=(50, 12)))
        assert np.allclose(covariance, covariance.T)


class TestConstraints:
    def test_max_weight_per_name_caps_and_reallocates(self, scores, context):
        weights = TopNEqualWeight(n=2).target_weights(scores, context)
        capped = MaxWeightPerName(limit=0.3).apply(weights, context)
        assert capped.abs().max().max() <= 0.3 + 1e-12

    def test_max_gross_scales_down(self, scores, context):
        weights = TopNEqualWeight(n=5, gross=2.0).target_weights(scores, context)
        limited = MaxGross(limit=1.0).apply(weights, context)
        assert limited.abs().sum(axis=1).max() == pytest.approx(1.0)

    def test_max_gross_leaves_a_compliant_book_alone(self, scores, context):
        weights = TopNEqualWeight(n=5, gross=0.5).target_weights(scores, context)
        pd.testing.assert_frame_equal(MaxGross(limit=1.0).apply(weights, context), weights)

    def test_max_net_bounds_directional_exposure(self, scores, context):
        weights = TopNEqualWeight(n=5).target_weights(scores, context)
        limited = MaxNet(limit=0.1).apply(weights, context)
        assert limited.sum(axis=1).abs().max() <= 0.1 + 1e-9

    def test_max_beta_bounds_portfolio_beta(self, scores, context):
        weights = TopNEqualWeight(n=5).target_weights(scores, context)
        limited = MaxBeta(limit=0.2).apply(weights, context)
        exposure = (limited * context.beta.reindex_like(limited)).sum(axis=1)
        assert exposure.abs().max() <= 0.2 + 1e-9

    def test_min_positions_blanks_a_thin_book(self, scores, context):
        weights = TopNEqualWeight(n=3).target_weights(scores, context)
        assert MinPositions(minimum=10).apply(weights, context).abs().to_numpy().max() == 0.0

    def test_max_group_weight_caps_a_sector(self, scores, context):
        context.groups = pd.Series({symbol: "tech" for symbol in SYMBOLS})
        weights = TopNEqualWeight(n=5).target_weights(scores, context)
        limited = MaxGroupWeight(limit=0.4).apply(weights, context)
        assert limited.sum(axis=1).max() <= 0.4 + 1e-9

    def test_chain_applies_in_order(self, scores, context):
        chain = ConstraintChain.of([MaxWeightPerName(0.2), MaxGross(0.5)])
        result = chain.apply(TopNEqualWeight(n=4).target_weights(scores, context), context)
        assert result.abs().sum(axis=1).max() == pytest.approx(0.5)

    def test_build_chain_from_a_mapping(self):
        chain = build_chain({"max_weight_per_name": 0.05, "min_positions": 10})
        assert len(chain.constraints) == 2
        with pytest.raises(ValueError, match="unknown constraint"):
            build_chain({"teleport": 1})


class TestSchedule:
    def test_weekly_rebalance_is_sparser_than_daily(self, dates):
        assert len(rebalance_dates(dates, "weekly")) < len(rebalance_dates(dates, "daily"))

    def test_rebalance_dates_are_real_sessions(self, dates):
        assert set(rebalance_dates(dates, "monthly")).issubset(set(dates))

    def test_unknown_frequency_is_rejected(self, dates):
        with pytest.raises(ValueError, match="unknown rebalance frequency"):
            rebalance_dates(dates, "hourly")

    def test_no_trade_band_suppresses_small_moves(self):
        index = pd.date_range("2024-01-01", periods=3, tz="UTC")
        targets = pd.DataFrame({"A": [0.10, 0.101, 0.30]}, index=index)
        held = no_trade_band(targets, epsilon=0.01)
        assert held["A"].tolist() == pytest.approx([0.10, 0.10, 0.30])

    def test_turnover_cap_limits_the_step(self):
        index = pd.date_range("2024-01-01", periods=2, tz="UTC")
        targets = pd.DataFrame({"A": [1.0, -1.0], "B": [0.0, 0.0]}, index=index)
        limited = limit_turnover(targets, maximum=0.5)
        assert limited["A"].iloc[0] == pytest.approx(0.5)

    def test_build_targets_marks_holds_as_nan(self, scores, context):
        targets = build_targets(scores, context, QuantileLongShort(), rebalance="monthly")
        assert targets.index.equals(scores.index)
        assert targets.isna().all(axis=1).sum() > 0

    def test_build_targets_is_deterministic(self, scores, context):
        first = build_targets(scores, context, QuantileLongShort(), rebalance="weekly")
        second = build_targets(scores, context, QuantileLongShort(), rebalance="weekly")
        pd.testing.assert_frame_equal(first, second)


def test_normalize_gross_hits_the_target():
    frame = pd.DataFrame({"A": [1.0, 2.0], "B": [-3.0, 0.0]})
    assert normalize_gross(frame, 2.0).abs().sum(axis=1).tolist() == pytest.approx([2.0, 2.0])
