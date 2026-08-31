"""Evaluation: metrics checked against closed-form answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Evaluation.metrics import (
    alpha_beta,
    annual_volatility,
    cagr,
    calmar,
    conditional_var,
    deflated_sharpe,
    downside_deviation,
    drawdown,
    drawdown_duration,
    information_ratio,
    max_drawdown,
    monthly_returns,
    performance,
    return_stats,
    sharpe,
    sortino,
    total_return,
    trading_stats,
    value_at_risk,
)
from Evaluation.report import compare, evaluate, load_run, report


def curve(values: list[float]) -> pd.Series:
    index = pd.bdate_range("2024-01-01", periods=len(values), tz="UTC", name="date")
    return pd.Series(values, index=index, name="equity")


class TestReturnMetrics:
    def test_total_return_by_hand(self):
        assert total_return(curve([100.0, 110.0, 121.0])) == pytest.approx(0.21)

    def test_cagr_of_a_doubling_over_one_year(self):
        equity = curve([100.0 * 2 ** (i / 252) for i in range(253)])
        assert cagr(equity) == pytest.approx(1.0, rel=1e-3)

    def test_flat_equity_has_zero_return(self):
        assert total_return(curve([100.0] * 10)) == 0.0
        assert cagr(curve([100.0] * 10)) == pytest.approx(0.0)


class TestRiskMetrics:
    def test_max_drawdown_by_hand(self):
        assert max_drawdown(curve([100.0, 120.0, 60.0, 90.0])) == pytest.approx(-0.5)

    def test_drawdown_series_is_zero_at_new_highs(self):
        under = drawdown(curve([100.0, 110.0, 120.0]))
        assert under.tolist() == pytest.approx([0.0, 0.0, 0.0])

    def test_drawdown_duration_counts_bars_underwater(self):
        assert drawdown_duration(curve([100.0, 90.0, 95.0, 99.0, 101.0])) == 3

    def test_volatility_scales_by_root_252(self):
        returns = pd.Series([0.01, -0.01] * 50)
        assert annual_volatility(returns) == pytest.approx(returns.std(ddof=1) * np.sqrt(252))

    def test_downside_deviation_ignores_gains(self):
        gains_only = pd.Series([0.01] * 20)
        assert downside_deviation(gains_only) == pytest.approx(0.0)

    def test_var_and_cvar_are_ordered(self):
        returns = pd.Series(np.random.default_rng(0).normal(0.0, 0.02, 1000))
        assert conditional_var(returns, 0.95) <= value_at_risk(returns, 0.95)


class TestRiskAdjusted:
    def test_sharpe_of_a_constant_return(self):
        assert sharpe(pd.Series([0.001] * 50)) == 0.0

    def test_sharpe_by_hand(self):
        returns = pd.Series([0.01, -0.005] * 100)
        expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
        assert sharpe(returns) == pytest.approx(expected)

    def test_sortino_exceeds_sharpe_for_right_skewed_returns(self):
        returns = pd.Series([0.03] * 30 + [-0.005] * 70)
        assert sortino(returns) > sharpe(returns)

    def test_calmar_is_cagr_over_drawdown(self):
        equity = curve([100.0, 120.0, 60.0, 130.0])
        assert calmar(equity) == pytest.approx(cagr(equity) / 0.5)

    def test_deflated_sharpe_falls_as_trials_rise(self):
        returns = pd.Series(np.random.default_rng(1).normal(0.0008, 0.01, 1000))
        assert deflated_sharpe(returns, 1) > deflated_sharpe(returns, 500)

    def test_deflated_sharpe_is_a_probability(self):
        returns = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 500))
        value = deflated_sharpe(returns, 20)
        assert 0.0 <= value <= 1.0


class TestAttribution:
    def test_beta_of_a_doubled_benchmark_is_two(self):
        index = pd.bdate_range("2024-01-01", periods=200, tz="UTC")
        market = pd.Series(np.random.default_rng(3).normal(0.0, 0.01, 200), index=index)
        alpha, beta = alpha_beta(2.0 * market, market)
        assert beta == pytest.approx(2.0)
        assert alpha == pytest.approx(0.0, abs=1e-9)

    def test_information_ratio_of_a_tracking_portfolio_is_zero(self):
        index = pd.bdate_range("2024-01-01", periods=100, tz="UTC")
        market = pd.Series(np.random.default_rng(4).normal(0.0, 0.01, 100), index=index)
        assert information_ratio(market, market) == 0.0


class TestTrading:
    def test_no_trades_gives_zero_turnover(self):
        equity = pd.DataFrame({"equity": [100.0, 101.0]},
                              index=pd.bdate_range("2024-01-01", periods=2, tz="UTC"))
        assert trading_stats(pd.DataFrame(), equity)["annual_turnover"] == 0.0

    def test_costs_are_summed(self):
        dates = pd.bdate_range("2024-01-01", periods=2, tz="UTC")
        equity = pd.DataFrame({"equity": [1_000_000.0, 1_000_000.0]}, index=dates)
        trades = pd.DataFrame(
            {
                "date": [dates[0], dates[1]],
                "symbol": ["A", "A"],
                "qty": [100.0, -100.0],
                "notional": [10_000.0, 10_000.0],
                "commission": [1.0, 1.0],
                "spread": [2.0, 2.0],
                "slippage": [3.0, 3.0],
            }
        )
        stats = trading_stats(trades, equity)
        assert stats["total_cost"] == pytest.approx(12.0)
        assert stats["annual_turnover"] == pytest.approx(0.02 * 252 / 2)

    def test_hit_rate_and_profit_factor(self):
        returns = pd.Series([0.02, -0.01, 0.02, -0.01])
        stats = return_stats(returns)
        assert stats["hit_rate"] == pytest.approx(0.5)
        assert stats["profit_factor"] == pytest.approx(2.0)


class TestPerformance:
    def test_headline_block_is_complete(self):
        index = pd.bdate_range("2024-01-01", periods=300, tz="UTC")
        equity_values = 1_000_000.0 * np.exp(
            np.cumsum(np.random.default_rng(5).normal(0.0004, 0.01, 300))
        )
        equity = pd.DataFrame({"equity": equity_values}, index=index)
        equity["leverage"] = 1.0
        equity["net"] = equity["equity"]

        summary = performance(equity)
        for key in ("total_return", "cagr", "sharpe", "sortino", "calmar", "max_drawdown",
                    "var_95", "cvar_95", "worst_month", "deflated_sharpe"):
            assert key in summary

    def test_monthly_returns_compound(self):
        index = pd.bdate_range("2024-01-01", periods=44, tz="UTC")
        returns = pd.Series(0.01, index=index)
        monthly = monthly_returns(returns)
        assert monthly.iloc[0] == pytest.approx(1.01 ** (index.month == 1).sum() - 1.0)


class TestReportPipeline:
    def test_report_is_written_for_a_saved_run(self, isolated_settings, tmp_path):
        from Strategy.BackTest import ExecutionConfig
        from Strategy.BackTest.costs import CostModel
        from Strategy.BackTest.vectorised import run as run_vector
        from Strategy.test.test_offline import flat_market, instruction

        data = flat_market({"AAA": list(np.linspace(100.0, 140.0, 300)), "BBB": [50.0] * 300})
        result = run_vector(
            instruction(data, {"AAA": 1.0}),
            data,
            CostModel(),
            ExecutionConfig(fill_price="close", execution_lag=1, participation_rate=1.0),
            name="report_test",
        )
        path = result.save()

        run = load_run(path)
        summary = evaluate(run)
        assert summary["total_return"] > 0.0

        html = report(run)
        assert html.exists()
        assert "Equity curve" in html.read_text(encoding="utf-8")

        table = compare([path])
        assert len(table) == 1
