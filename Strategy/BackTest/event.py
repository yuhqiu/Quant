"""Event-driven backtest engine.

Slower than the vectorised engine and deliberately so: it walks orders one at a
time, which is what path-dependent logic such as stops and trailing exits needs.
With no risk rules configured it must reproduce the vectorised result exactly,
and that equivalence is itself a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from Common.logging import get_logger

from .costs import CostModel
from .engine import ExecutionConfig, MarketData, TradeLog, lagged_targets, positions_frame
from .result import BacktestResult, build_equity_frame, make_run_id

log = get_logger(__name__)


@dataclass(frozen=True)
class RiskRules:
    """Path-dependent exits. All fractions of price, all optional."""

    stop_loss: float | None = None
    trailing_stop: float | None = None
    take_profit: float | None = None

    @property
    def active(self) -> bool:
        return any((self.stop_loss, self.trailing_stop, self.take_profit))


@dataclass
class Position:
    index: int
    symbol: str
    shares: float = 0.0
    entry_price: float = 0.0
    peak_price: float = 0.0
    trough_price: float = float("inf")

    def observe(self, price: float) -> None:
        self.peak_price = max(self.peak_price, price)
        self.trough_price = min(self.trough_price, price)


@dataclass
class Order:
    index: int
    symbol: str
    quantity: float
    price: float
    reason: str = "rebalance"


@dataclass
class Book:
    positions: dict[int, Position] = field(default_factory=dict)

    def get(self, index: int, symbol: str) -> Position:
        if index not in self.positions:
            self.positions[index] = Position(index, symbol)
        return self.positions[index]

    def shares(self, index: int) -> float:
        position = self.positions.get(index)
        return position.shares if position else 0.0

    def held(self) -> list[Position]:
        return [p for p in self.positions.values() if p.shares != 0.0]


def run(
    targets: pd.DataFrame,
    data: MarketData,
    costs: CostModel | None = None,
    config: ExecutionConfig | None = None,
    risk: RiskRules | None = None,
    name: str = "strategy",
    run_id: str | None = None,
    spec: dict | None = None,
) -> BacktestResult:
    costs = costs or CostModel()
    config = config or ExecutionConfig()
    risk = risk or RiskRules()

    periods, count = data.shape
    symbols = list(data.symbols)
    instructions = lagged_targets(targets, data.dates, data.symbols, config.execution_lag)
    fills = data.fill_prices(config.fill_price)

    book = Book()
    cash = float(config.initial_capital)
    last_price = np.zeros(count)

    equity_path = np.zeros(periods)
    cash_path = np.zeros(periods)
    long_path = np.zeros(periods)
    short_path = np.zeros(periods)
    share_history = np.zeros((periods, count))
    value_history = np.zeros((periods, count))
    adjustments = np.zeros(count)
    trades = TradeLog()

    for step in range(periods):
        close = data.close[step]
        date = data.dates[step]

        _apply_corporate_actions(book, data, step, adjustments)
        cash += _collect_dividends(book, data, step)
        cash += _close_delisted(book, data, step, close, last_price, trades)

        if risk.active:
            cash += _apply_risk(book, data, step, risk, costs, trades)

        mark = np.where(last_price > 0.0, last_price, np.nan_to_num(close))
        equity_start = cash + sum(p.shares * mark[p.index] for p in book.held())

        row = instructions[step]
        if not np.isnan(row).all():
            orders = _generate_orders(row, book, data, step, fills, symbols, equity_start, config)
            cash = _settle(orders, book, cash, data, step, costs, config, trades, date)

        priced = np.isfinite(close) & (close > 0.0)
        price_now = np.where(priced, close, last_price)

        short_value = -sum(min(p.shares, 0.0) * price_now[p.index] for p in book.held())
        cash -= costs.borrow(float(short_value))
        cash += costs.interest(cash)

        last_price = np.where(priced, close, last_price)
        long_value = 0.0
        short_value = 0.0
        for position in book.held():
            value = position.shares * last_price[position.index]
            share_history[step, position.index] = position.shares
            value_history[step, position.index] = value
            position.observe(last_price[position.index])
            if value >= 0.0:
                long_value += value
            else:
                short_value -= value

        equity_path[step] = cash + long_value - short_value
        cash_path[step] = cash
        long_path[step] = long_value
        short_path[step] = short_value

    equity = build_equity_frame(data.dates, equity_path, cash_path, long_path, short_path)
    positions = positions_frame(data.dates, data.symbols, share_history, value_history, equity_path)
    trade_frame = trades.to_frame()

    resolved_spec = dict(spec or {})
    resolved_spec.setdefault("name", name)
    from .vectorised import _headline

    return BacktestResult(
        name=name,
        run_id=run_id or make_run_id("event0000"),
        equity=equity,
        positions=positions,
        trades=trade_frame,
        spec=resolved_spec,
        metrics=_headline(equity, trade_frame, config),
        engine="event_driven",
        adjustments=pd.Series(adjustments, index=list(data.symbols)),
    )


def _apply_corporate_actions(
    book: Book, data: MarketData, step: int, adjustments: np.ndarray
) -> None:
    ratios = data.split_ratio[step]
    for position in book.held():
        ratio = ratios[position.index]
        if ratio > 0.0 and ratio != 1.0:
            before = position.shares
            position.shares *= ratio
            adjustments[position.index] += position.shares - before
            position.entry_price /= ratio
            position.peak_price /= ratio


def _collect_dividends(book: Book, data: MarketData, step: int) -> float:
    dividends = data.dividend[step]
    return float(sum(p.shares * dividends[p.index] for p in book.held()))


def _close_delisted(
    book: Book,
    data: MarketData,
    step: int,
    close: np.ndarray,
    last_price: np.ndarray,
    trades: TradeLog,
) -> float:
    proceeds = 0.0
    for position in list(book.held()):
        index = position.index
        if np.isfinite(close[index]) and close[index] > 0.0:
            continue
        if last_price[index] <= 0.0:
            continue
        quantity = -position.shares
        price = float(last_price[index])
        proceeds += -quantity * price
        trades.add(
            data.dates[step],
            np.array([position.symbol], dtype=object),
            np.array([quantity]),
            np.array([price]),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
        )
        position.shares = 0.0
    return proceeds


def _apply_risk(
    book: Book,
    data: MarketData,
    step: int,
    risk: RiskRules,
    costs: CostModel,
    trades: TradeLog,
) -> float:
    """Stops are checked against the bar's own low and high, and fill at the trigger."""
    proceeds = 0.0
    low, high = data.low[step], data.high[step]

    for position in list(book.held()):
        index = position.index
        if not (np.isfinite(low[index]) and np.isfinite(high[index])):
            continue

        long_side = position.shares > 0.0
        trigger: float | None = None

        if risk.stop_loss and position.entry_price > 0.0:
            level = position.entry_price * (
                1.0 - risk.stop_loss if long_side else 1.0 + risk.stop_loss
            )
            if (long_side and low[index] <= level) or (not long_side and high[index] >= level):
                trigger = level

        if trigger is None and risk.trailing_stop:
            reference = position.peak_price if long_side else position.trough_price
            if np.isfinite(reference) and reference > 0.0:
                level = reference * (
                    1.0 - risk.trailing_stop if long_side else 1.0 + risk.trailing_stop
                )
                if (long_side and low[index] <= level) or (not long_side and high[index] >= level):
                    trigger = level

        if trigger is None and risk.take_profit and position.entry_price > 0.0:
            level = position.entry_price * (
                1.0 + risk.take_profit if long_side else 1.0 - risk.take_profit
            )
            if (long_side and high[index] >= level) or (not long_side and low[index] <= level):
                trigger = level

        if trigger is None:
            continue

        quantity = -position.shares
        notional = np.array([abs(quantity) * trigger])
        commission = costs.commission(notional, np.array([quantity]))
        spread = costs.spread(notional)
        proceeds += -quantity * trigger - float(commission[0] + spread[0])
        trades.add(
            data.dates[step],
            np.array([position.symbol], dtype=object),
            np.array([quantity]),
            np.array([trigger]),
            commission,
            spread,
            np.zeros(1),
        )
        position.shares = 0.0
    return proceeds


def _generate_orders(
    row: np.ndarray,
    book: Book,
    data: MarketData,
    step: int,
    fills: np.ndarray,
    symbols: list[str],
    equity_start: float,
    config: ExecutionConfig,
) -> list[Order]:
    orders: list[Order] = []
    volume = data.volume[step]
    close = data.close[step]
    fill_row = fills[step]

    candidates = set(np.flatnonzero(np.nan_to_num(row) != 0.0).tolist())
    candidates.update(position.index for position in book.held())

    for index in sorted(candidates):
        price = fill_row[index]
        if not (np.isfinite(price) and price > 0.0):
            continue
        if not (np.isfinite(close[index]) and close[index] > 0.0):
            continue
        if not (np.isfinite(volume[index]) and volume[index] > 0.0):
            continue

        weight = row[index]
        weight = 0.0 if np.isnan(weight) else float(weight)
        desired = weight * equity_start / price
        if config.whole_shares:
            desired = float(np.trunc(desired))

        quantity = desired - book.shares(index)
        capacity = config.participation_rate * float(volume[index])
        quantity = float(np.clip(quantity, -capacity, capacity))
        if config.whole_shares:
            quantity = float(np.trunc(quantity))
        if quantity == 0.0:
            continue
        orders.append(Order(index, symbols[index], quantity, float(price)))
    return orders


def _settle(
    orders: list[Order],
    book: Book,
    cash: float,
    data: MarketData,
    step: int,
    costs: CostModel,
    config: ExecutionConfig,
    trades: TradeLog,
    date: pd.Timestamp,
) -> float:
    if not orders:
        return cash

    index = np.array([order.index for order in orders])
    quantity = np.array([order.quantity for order in orders])
    price = np.array([order.price for order in orders])
    charges = _charges(quantity, price, data, step, index, costs)

    if not config.allow_margin:
        proceeds = float(np.sum(np.where(quantity < 0.0, -quantity * price, 0.0)))
        outlay = float(np.sum(np.where(quantity > 0.0, quantity * price, 0.0)))
        buy_cost = float(np.sum(charges[quantity > 0.0]))
        sell_cost = float(np.sum(charges[quantity < 0.0]))
        available = cash + proceeds - sell_cost
        needed = outlay + buy_cost
        if needed > available and needed > 0.0:
            scale = max(min(available / needed, 1.0), 0.0)
            quantity = np.where(quantity > 0.0, quantity * scale, quantity)
            if config.whole_shares:
                quantity = np.trunc(quantity)

    keep = quantity != 0.0
    index, quantity, price = index[keep], quantity[keep], price[keep]
    if index.size == 0:
        return cash

    commission, spread, slippage = _components(quantity, price, data, step, index, costs)
    cash -= float(np.sum(quantity * price))
    cash -= float(np.sum(commission) + np.sum(spread) + np.sum(slippage))

    for position_index, filled, fill_price in zip(index, quantity, price, strict=True):
        symbol = data.symbols[position_index]
        position = book.get(int(position_index), symbol)
        before = position.shares
        position.shares += float(filled)
        if before == 0.0 or np.sign(before) != np.sign(position.shares):
            position.entry_price = float(fill_price)
            position.peak_price = float(fill_price)
            position.trough_price = float(fill_price)

    trades.add(
        date,
        np.asarray(data.symbols, dtype=object)[index],
        quantity,
        price,
        commission,
        spread,
        slippage,
    )
    return cash


def _components(
    quantity: np.ndarray,
    price: np.ndarray,
    data: MarketData,
    step: int,
    index: np.ndarray,
    costs: CostModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    notional = np.abs(quantity) * price
    commission = costs.commission(notional, quantity)
    spread = costs.spread(
        notional, data.high[step][index], data.low[step][index], data.close[step][index]
    )
    slippage = costs.slippage(notional, data.sigma[step][index], data.advd[step][index])
    return commission, spread, slippage


def _charges(
    quantity: np.ndarray,
    price: np.ndarray,
    data: MarketData,
    step: int,
    index: np.ndarray,
    costs: CostModel,
) -> np.ndarray:
    return sum(_components(quantity, price, data, step, index, costs))
