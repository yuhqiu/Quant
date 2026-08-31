"""Vectorised backtest engine.

One pass over the date axis with whole-universe NumPy operations on each bar.
Fast enough for parameter sweeps over thousands of symbols, and the primary
engine for daily cross-sectional strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Common.logging import get_logger

from .costs import CostModel
from .engine import ExecutionConfig, MarketData, TradeLog, lagged_targets, positions_frame
from .result import BacktestResult, build_equity_frame, make_run_id

log = get_logger(__name__)


def run(
    targets: pd.DataFrame,
    data: MarketData,
    costs: CostModel | None = None,
    config: ExecutionConfig | None = None,
    name: str = "strategy",
    run_id: str | None = None,
    spec: dict | None = None,
) -> BacktestResult:
    """Simulate ``targets`` against ``data`` and return the shared result format."""
    costs = costs or CostModel()
    config = config or ExecutionConfig()

    periods, count = data.shape
    symbol_array = np.asarray(data.symbols, dtype=object)
    instructions = lagged_targets(targets, data.dates, data.symbols, config.execution_lag)
    fills = data.fill_prices(config.fill_price)

    shares = np.zeros(count)
    cash = float(config.initial_capital)
    last_price = np.zeros(count)

    equity_path = np.zeros(periods)
    cash_path = np.zeros(periods)
    long_path = np.zeros(periods)
    short_path = np.zeros(periods)
    share_history = np.zeros((periods, count))
    value_history = np.zeros((periods, count))

    trades = TradeLog()
    carried = np.zeros(count)
    adjustments = np.zeros(count)

    for step in range(periods):
        close = data.close[step]
        priced = np.isfinite(close) & (close > 0.0)

        # --- corporate actions, before anything else touches the position ---
        split = data.split_ratio[step]
        splitting = (split > 0.0) & (split != 1.0)
        if splitting.any():
            adjusted = np.where(splitting, shares * split, shares)
            adjustments += adjusted - shares
            shares = adjusted
            carried = np.where(splitting, carried * split, carried)

        dividend = data.dividend[step]
        paying = (dividend != 0.0) & (shares != 0.0)
        if paying.any():
            cash += float(np.sum(shares[paying] * dividend[paying]))

        # --- delisting sweep: a name that stops printing is closed out ---
        gone = (~priced) & (shares != 0.0) & (last_price > 0.0)
        if gone.any():
            index = np.flatnonzero(gone)
            quantity = -shares[index]
            price = last_price[index]
            cash += float(np.sum(-quantity * price))
            trades.add(
                data.dates[step], symbol_array[index], quantity, price,
                np.zeros(index.size), np.zeros(index.size), np.zeros(index.size),
            )
            shares[index] = 0.0
            carried[index] = 0.0

        mark = np.where(last_price > 0.0, last_price, np.where(priced, close, 0.0))
        equity_start = cash + float(np.dot(shares, mark))

        # --- orders ---
        row = instructions[step]
        wanted = ~np.isnan(row).all()
        if wanted or (config.carry_unfilled and np.any(carried != 0.0)):
            fill = fills[step]
            tradable = (
                priced
                & np.isfinite(fill)
                & (fill > 0.0)
                & np.isfinite(data.volume[step])
                & (data.volume[step] > 0.0)
            )
            if wanted:
                weights = np.nan_to_num(row)
                desired = np.where(tradable, weights * equity_start / np.where(fill > 0.0, fill, 1.0), shares)
                if config.whole_shares:
                    desired = np.trunc(desired)
                quantity = desired - shares
            else:
                quantity = np.zeros(count)

            if config.carry_unfilled:
                quantity = quantity + carried
            quantity = np.where(tradable, quantity, 0.0)

            capacity = config.participation_rate * np.nan_to_num(data.volume[step])
            requested = quantity.copy()
            quantity = np.clip(quantity, -capacity, capacity)
            carried = (requested - quantity) if config.carry_unfilled else np.zeros(count)

            if config.whole_shares:
                quantity = np.trunc(quantity)

            active = quantity != 0.0
            if active.any():
                cash = _execute(
                    cash, shares, quantity, active, fill, data, step, costs, config, trades
                )

        # --- financing ---
        price_now = np.where(priced, close, last_price)
        short_value = float(-np.sum(np.minimum(shares, 0.0) * price_now))
        cash -= costs.borrow(short_value)
        cash += costs.interest(cash)

        # --- mark to market ---
        last_price = np.where(priced, close, last_price)
        values = shares * last_price
        long_value = float(np.sum(np.maximum(values, 0.0)))
        short_value = float(-np.sum(np.minimum(values, 0.0)))

        equity_path[step] = cash + long_value - short_value
        cash_path[step] = cash
        long_path[step] = long_value
        short_path[step] = short_value
        share_history[step] = shares
        value_history[step] = values

    equity = build_equity_frame(data.dates, equity_path, cash_path, long_path, short_path)
    positions = positions_frame(data.dates, data.symbols, share_history, value_history, equity_path)
    trade_frame = trades.to_frame()

    resolved_spec = dict(spec or {})
    resolved_spec.setdefault("name", name)
    metrics = _headline(equity, trade_frame, config)

    return BacktestResult(
        name=name,
        run_id=run_id or make_run_id("vector000"),
        equity=equity,
        positions=positions,
        trades=trade_frame,
        spec=resolved_spec,
        metrics=metrics,
        engine="vectorised",
        adjustments=pd.Series(adjustments, index=list(data.symbols)),
    )


def _execute(
    cash: float,
    shares: np.ndarray,
    quantity: np.ndarray,
    active: np.ndarray,
    fill: np.ndarray,
    data: MarketData,
    step: int,
    costs: CostModel,
    config: ExecutionConfig,
    trades: TradeLog,
) -> float:
    """Price the order book, enforce the cash constraint, settle and record."""
    index = np.flatnonzero(active)
    order = quantity[index]
    price = fill[index]

    charges = _charge(order, price, data, step, index, costs)
    proceeds = float(np.sum(np.where(order < 0.0, -order * price, 0.0)))
    outlay = float(np.sum(np.where(order > 0.0, order * price, 0.0)))

    if not config.allow_margin:
        buy_cost = float(np.sum(charges[order > 0.0]))
        sell_cost = float(np.sum(charges[order < 0.0]))
        available = cash + proceeds - sell_cost
        needed = outlay + buy_cost
        if needed > available and needed > 0.0:
            # Scale the buy side back to what the cash actually supports.
            scale = max(min(available / needed, 1.0), 0.0)
            order = np.where(order > 0.0, order * scale, order)
            if config.whole_shares:
                order = np.trunc(order)
            charges = _charge(order, price, data, step, index, costs)
            outlay = float(np.sum(np.where(order > 0.0, order * price, 0.0)))

    settled = order != 0.0
    index, order, price = index[settled], order[settled], price[settled]

    cash -= float(np.sum(order * price))
    commission, spread, slippage = _components(order, price, data, step, index, costs)
    cash -= float(np.sum(commission) + np.sum(spread) + np.sum(slippage))
    shares[index] += order

    trades.add(
        data.dates[step],
        np.asarray(data.symbols, dtype=object)[index],
        order,
        price,
        commission,
        spread,
        slippage,
    )
    return cash


def _components(
    order: np.ndarray,
    price: np.ndarray,
    data: MarketData,
    step: int,
    index: np.ndarray,
    costs: CostModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    notional = np.abs(order) * price
    commission = costs.commission(notional, order)
    spread = costs.spread(
        notional, data.high[step][index], data.low[step][index], data.close[step][index]
    )
    slippage = costs.slippage(notional, data.sigma[step][index], data.advd[step][index])
    return commission, spread, slippage


def _charge(
    order: np.ndarray,
    price: np.ndarray,
    data: MarketData,
    step: int,
    index: np.ndarray,
    costs: CostModel,
) -> np.ndarray:
    return sum(_components(order, price, data, step, index, costs))


def _headline(equity: pd.DataFrame, trades: pd.DataFrame, config: ExecutionConfig) -> dict:
    returns = equity["ret"]
    years = max((len(equity) - 1) / 252.0, 1e-9)
    total = float(equity["equity"].iloc[-1] / config.initial_capital - 1.0)
    volatility = float(returns.std() * np.sqrt(252.0))
    drawdown = equity["equity"] / equity["equity"].cummax() - 1.0

    return {
        "initial_capital": config.initial_capital,
        "final_equity": float(equity["equity"].iloc[-1]),
        "total_return": total,
        "cagr": float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1.0 else -1.0,
        "volatility": volatility,
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(252.0)) if returns.std() else 0.0,
        "max_drawdown": float(drawdown.min()),
        "bars": int(len(equity)),
        "trades": int(len(trades)),
        "total_costs": float(
            trades[["commission", "spread", "slippage"]].to_numpy().sum()
        ) if not trades.empty else 0.0,
    }
