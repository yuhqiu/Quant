"""Backtest correctness: invariants, a hand-computed scenario, engine equivalence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Portfolio import build_targets
from Portfolio.base import MarketContext
from Portfolio.constructors import QuantileLongShort
from Strategy.BackTest import ExecutionConfig, MarketData, run
from Strategy.BackTest.costs import ZERO_COST, CostModel
from Strategy.BackTest.event import RiskRules
from Strategy.BackTest.event import run as run_event
from Strategy.BackTest.vectorised import run as run_vector
from Strategy.spec import StrategySpec
from Strategy.walkforward import in_sample_split, walk_forward

SYMBOLS = ("AAA", "BBB")


def flat_market(
    closes: dict[str, list[float]],
    volume: float = 1_000_000.0,
    dividend: dict[str, list[float]] | None = None,
    split: dict[str, list[float]] | None = None,
) -> MarketData:
    """A market with no intrabar range: open == high == low == close."""
    periods = len(next(iter(closes.values())))
    dates = pd.bdate_range("2024-01-02", periods=periods, tz="UTC", name="date")
    price = pd.DataFrame(closes, index=dates, dtype="float64")

    frames = {
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": pd.DataFrame(volume, index=dates, columns=price.columns),
        "dividend": pd.DataFrame(dividend or 0.0, index=dates, columns=price.columns),
        "split_ratio": pd.DataFrame(split or 0.0, index=dates, columns=price.columns),
        "vol_20d": pd.DataFrame(0.2, index=dates, columns=price.columns),
        "advd_20": pd.DataFrame(1e9, index=dates, columns=price.columns),
    }
    return MarketData.from_frames(frames, dates, list(price.columns))


def instruction(data: MarketData, weights: dict[str, float], row: int = 0) -> pd.DataFrame:
    targets = pd.DataFrame(np.nan, index=data.dates, columns=list(data.symbols))
    targets.iloc[row] = [weights.get(symbol, 0.0) for symbol in data.symbols]
    return targets


class TestKnownAnswer:
    """A two-symbol, ten-bar scenario whose equity path is written out by hand."""

    def test_flat_price_leaves_equity_unchanged(self):
        data = flat_market({"AAA": [100.0] * 10, "BBB": list(np.arange(100.0, 110.0))})
        result = run_vector(
            instruction(data, {"AAA": 1.0}),
            data,
            ZERO_COST,
            ExecutionConfig(fill_price="close", execution_lag=1, participation_rate=1.0),
        )
        # 1,000,000 / 100 = 10,000 whole shares, no cash left over, price never moves.
        assert result.equity["equity"].tolist() == pytest.approx([1_000_000.0] * 10)
        assert result.positions.query("symbol == 'AAA'")["shares"].max() == 10_000.0

    def test_rising_price_equity_path_by_hand(self):
        data = flat_market({"AAA": [100.0] * 10, "BBB": list(np.arange(100.0, 110.0))})
        result = run_vector(
            instruction(data, {"BBB": 1.0}),
            data,
            ZERO_COST,
            ExecutionConfig(fill_price="close", execution_lag=1, participation_rate=1.0),
        )
        # Bar 1 fills at 101: trunc(1,000,000 / 101) = 9,900 shares, 100 cash left.
        expected = [1_000_000.0] + [100.0 + 9_900.0 * price for price in np.arange(101.0, 110.0)]
        assert result.equity["equity"].tolist() == pytest.approx(expected)
        assert result.equity["cash"].iloc[-1] == pytest.approx(100.0)

    def test_dividend_is_credited_on_the_ex_date(self):
        dividends = {"AAA": [0.0] * 5 + [2.0] + [0.0] * 4, "BBB": [0.0] * 10}
        data = flat_market({"AAA": [100.0] * 10, "BBB": [100.0] * 10}, dividend=dividends)
        result = run_vector(
            instruction(data, {"AAA": 1.0}),
            data,
            ZERO_COST,
            ExecutionConfig(fill_price="close", execution_lag=1, participation_rate=1.0),
        )
        # 10,000 shares times $2 = $20,000 of cash on bar 5, and it stays there.
        assert result.equity["equity"].iloc[-1] == pytest.approx(1_020_000.0)
        assert result.equity["cash"].iloc[-1] == pytest.approx(20_000.0)

    def test_split_leaves_equity_unchanged(self):
        prices = [100.0] * 5 + [50.0] * 5
        splits = {"AAA": [0.0] * 5 + [2.0] + [0.0] * 4, "BBB": [0.0] * 10}
        data = flat_market({"AAA": prices, "BBB": [100.0] * 10}, split=splits)
        result = run_vector(
            instruction(data, {"AAA": 1.0}),
            data,
            ZERO_COST,
            ExecutionConfig(fill_price="close", execution_lag=1, participation_rate=1.0),
        )
        assert result.equity["equity"].tolist() == pytest.approx([1_000_000.0] * 10)

    def test_zero_signal_returns_exactly_the_cash_rate(self):
        data = flat_market({"AAA": [100.0] * 10, "BBB": [100.0] * 10})
        targets = pd.DataFrame(0.0, index=data.dates, columns=list(data.symbols))
        result = run_vector(
            targets,
            data,
            CostModel(cash_rate_annual=0.0252, commission_bps=0.0, half_spread_bps=0.0,
                      estimate_spread_from_range=False, impact_coefficient=0.0),
            ExecutionConfig(fill_price="close", execution_lag=1),
        )
        daily = 0.0252 / 252.0
        assert result.equity["equity"].iloc[-1] == pytest.approx(1_000_000.0 * (1 + daily) ** 10)
        assert result.trades.empty

    def test_costs_are_deducted(self):
        data = flat_market({"AAA": [100.0] * 10, "BBB": [100.0] * 10})
        config = ExecutionConfig(fill_price="close", execution_lag=1, participation_rate=1.0)
        free = run_vector(instruction(data, {"AAA": 1.0}), data, ZERO_COST, config)
        charged = run_vector(instruction(data, {"AAA": 1.0}), data, CostModel(), config)
        assert charged.final_equity < free.final_equity
        assert charged.cost_summary()["total"] > 0.0


class TestInvariants:
    @pytest.fixture
    def result(self):
        generator = np.random.default_rng(4)
        prices = {
            symbol: list(100.0 * np.exp(np.cumsum(generator.normal(0.0005, 0.02, 120))))
            for symbol in [f"S{i:02d}" for i in range(12)]
        }
        data = flat_market(prices)
        scores = pd.DataFrame(
            generator.normal(size=(len(data.dates), len(data.symbols))),
            index=data.dates,
            columns=list(data.symbols),
        )
        context = MarketContext(prices=pd.DataFrame(
            data.close, index=data.dates, columns=list(data.symbols)
        ))
        targets = build_targets(scores, context, QuantileLongShort(quantiles=3), rebalance="weekly")
        return run_vector(targets, data, CostModel(), ExecutionConfig(participation_rate=1.0))

    def test_equity_equals_cash_plus_positions(self, result):
        assert result.check_invariants() == []

    def test_trades_reconcile_position_deltas(self, result):
        traded = result.trades.groupby("symbol")["qty"].sum()
        final = (
            result.positions[result.positions["date"] == result.positions["date"].max()]
            .set_index("symbol")["shares"]
        )
        for symbol, quantity in traded.items():
            assert final.get(symbol, 0.0) == pytest.approx(quantity, abs=1e-9)

    def test_cash_never_goes_negative_without_margin(self, result):
        assert result.equity["cash"].min() >= -1e-6

    def test_equity_is_finite_everywhere(self, result):
        assert np.isfinite(result.equity["equity"].to_numpy()).all()

    def test_trades_exist_and_are_priced(self, result):
        assert not result.trades.empty
        assert (result.trades["price"] > 0).all()


class TestExecutionAssumptions:
    def test_execution_lag_delays_the_fill(self):
        data = flat_market({"AAA": [100.0] * 6, "BBB": [100.0] * 6})
        config = ExecutionConfig(fill_price="close", execution_lag=3, participation_rate=1.0)
        result = run_vector(instruction(data, {"AAA": 1.0}), data, ZERO_COST, config)
        first_trade = result.trades["date"].min()
        assert first_trade == data.dates[3]

    def test_participation_cap_limits_the_fill(self):
        data = flat_market({"AAA": [100.0] * 6, "BBB": [100.0] * 6}, volume=1_000.0)
        config = ExecutionConfig(fill_price="close", execution_lag=1, participation_rate=0.1)
        result = run_vector(instruction(data, {"AAA": 1.0}), data, ZERO_COST, config)
        assert result.trades["qty"].abs().max() == pytest.approx(100.0)

    def test_no_trading_on_a_bar_with_no_volume(self):
        data = flat_market({"AAA": [100.0] * 6, "BBB": [100.0] * 6}, volume=0.0)
        result = run_vector(
            instruction(data, {"AAA": 1.0}), data, ZERO_COST,
            ExecutionConfig(fill_price="close", execution_lag=1),
        )
        assert result.trades.empty

    def test_fill_price_modes_differ(self):
        dates = pd.bdate_range("2024-01-02", periods=5, tz="UTC")
        columns = ["AAA"]
        frames = {
            "open": pd.DataFrame(100.0, index=dates, columns=columns),
            "high": pd.DataFrame(110.0, index=dates, columns=columns),
            "low": pd.DataFrame(90.0, index=dates, columns=columns),
            "close": pd.DataFrame(105.0, index=dates, columns=columns),
            "volume": pd.DataFrame(1e9, index=dates, columns=columns),
            "advd_20": pd.DataFrame(1e12, index=dates, columns=columns),
        }
        data = MarketData.from_frames(frames, dates, columns)
        assert data.fill_prices("next_open")[0][0] == 100.0
        assert data.fill_prices("close")[0][0] == 105.0
        assert data.fill_prices("vwap_proxy")[0][0] == pytest.approx((110 + 90 + 105) / 3)

    def test_margin_is_off_by_default(self):
        data = flat_market({"AAA": [100.0] * 6, "BBB": [100.0] * 6})
        result = run_vector(
            instruction(data, {"AAA": 3.0}), data, ZERO_COST,
            ExecutionConfig(fill_price="close", execution_lag=1, participation_rate=1.0),
        )
        assert result.equity["cash"].min() >= -1e-6


class TestEngineEquivalence:
    @pytest.fixture
    def scenario(self):
        generator = np.random.default_rng(11)
        symbols = [f"S{i:02d}" for i in range(8)]
        prices = {
            symbol: list(100.0 * np.exp(np.cumsum(generator.normal(0.0004, 0.015, 90))))
            for symbol in symbols
        }
        data = flat_market(prices)
        scores = pd.DataFrame(
            generator.normal(size=(len(data.dates), len(symbols))),
            index=data.dates,
            columns=symbols,
        )
        context = MarketContext(
            prices=pd.DataFrame(data.close, index=data.dates, columns=symbols)
        )
        targets = build_targets(scores, context, QuantileLongShort(quantiles=2), rebalance="weekly")
        return targets, data

    def test_both_engines_agree_on_equity(self, scenario):
        targets, data = scenario
        config = ExecutionConfig(fill_price="close", execution_lag=1, participation_rate=1.0)
        vector = run_vector(targets, data, CostModel(), config)
        events = run_event(targets, data, CostModel(), config, RiskRules())
        pd.testing.assert_series_equal(
            vector.equity["equity"], events.equity["equity"], rtol=1e-9, check_names=False
        )

    def test_both_engines_agree_on_trades(self, scenario):
        targets, data = scenario
        config = ExecutionConfig(fill_price="close", execution_lag=1, participation_rate=1.0)
        vector = run_vector(targets, data, CostModel(), config).trades
        events = run_event(targets, data, CostModel(), config).trades
        assert len(vector) == len(events)
        assert vector["qty"].sum() == pytest.approx(events["qty"].sum())

    def test_dispatch_selects_the_engine(self, scenario):
        targets, data = scenario
        assert run(targets, data, engine="vectorised").engine == "vectorised"
        assert run(targets, data, engine="event_driven").engine == "event_driven"
        with pytest.raises(ValueError, match="unknown engine"):
            run(targets, data, engine="quantum")

    def test_stops_make_the_event_engine_diverge(self, scenario):
        """The stop rule must actually do something, or the engine is pointless."""
        targets, data = scenario
        config = ExecutionConfig(fill_price="close", execution_lag=1, participation_rate=1.0)
        plain = run_event(targets, data, CostModel(), config, RiskRules())
        stopped = run_event(targets, data, CostModel(), config, RiskRules(stop_loss=0.02))
        assert plain.final_equity != stopped.final_equity


class TestDeterminism:
    def test_the_same_inputs_produce_the_same_outputs(self):
        data = flat_market({"AAA": [100.0, 101.0, 102.0, 103.0], "BBB": [50.0] * 4})
        targets = instruction(data, {"AAA": 0.5, "BBB": 0.5})
        config = ExecutionConfig(fill_price="close", execution_lag=1, participation_rate=1.0)
        first = run_vector(targets, data, CostModel(), config, run_id="fixed")
        second = run_vector(targets, data, CostModel(), config, run_id="fixed")
        pd.testing.assert_frame_equal(first.equity, second.equity)
        pd.testing.assert_frame_equal(first.trades, second.trades)


class TestSpec:
    def test_library_specs_load(self, project_root):
        for path in sorted((project_root / "Strategy" / "Library").glob("*.toml")):
            spec = StrategySpec.load(path)
            assert spec.name
            assert spec.build_signal() is not None
            assert spec.build_constructor() is not None

    def test_hash_ignores_the_window_but_not_the_parameters(self, project_root):
        spec = StrategySpec.load(project_root / "Strategy" / "Library" / "momentum.toml")
        assert spec.hash == spec.with_overrides(start="1999-01-01").hash
        assert spec.hash != spec.with_overrides(rebalance="daily").hash

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValueError, match="unknown strategy fields"):
            StrategySpec.from_dict({"name": "x", "leverage_to_the_moon": 10})

    def test_costs_are_on_unless_switched_off(self):
        assert StrategySpec(name="x").costs.enabled is True


class TestWalkForward:
    def test_windows_do_not_overlap_and_respect_the_gap(self):
        dates = pd.bdate_range("2015-01-01", periods=2000, tz="UTC")
        splits = walk_forward(dates, train_size=500, test_size=250, purge=21, embargo=5)
        assert splits
        for split in splits:
            gap = (dates.get_loc(split.test_start) - dates.get_loc(split.train_end)) - 1
            assert gap == 26
            assert split.train_end < split.test_start <= split.test_end

    def test_anchored_windows_share_a_start(self):
        dates = pd.bdate_range("2015-01-01", periods=2000, tz="UTC")
        splits = walk_forward(dates, 500, 250, anchored=True)
        assert len({split.train_start for split in splits}) == 1

    def test_in_sample_split_is_ordered(self):
        dates = pd.bdate_range("2015-01-01", periods=1000, tz="UTC")
        train_end, test_start = in_sample_split(dates, 0.7)
        assert train_end < test_start


class TestOosLedger:
    def test_touches_are_counted(self, isolated_settings):
        from Strategy.oos import record_touch, touches

        assert touches("abc") == 0
        record_touch("abc", "momentum", "2020-2024")
        record_touch("abc", "momentum", "2020-2024")
        assert touches("abc") == 2


class TestPersistence:
    def test_result_round_trips_through_disk(self, isolated_settings):
        data = flat_market({"AAA": [100.0, 101.0, 102.0], "BBB": [50.0] * 3})
        result = run_vector(
            instruction(data, {"AAA": 1.0}), data, CostModel(),
            ExecutionConfig(fill_price="close", execution_lag=1, participation_rate=1.0),
            name="unit_test",
        )
        from Strategy.BackTest.result import BacktestResult

        path = result.save()
        restored = BacktestResult.load(path)
        pd.testing.assert_series_equal(
            result.equity["equity"], restored.equity["equity"], check_freq=False
        )
        assert restored.name == "unit_test"
