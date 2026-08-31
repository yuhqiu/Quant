"""Rebalance timing, no-trade bands and turnover limits.

Weights are decided on bar ``T`` from information available at ``T``; the backtest
applies them at ``T + execution_lag``. Dates that are not rebalance dates carry
NaN, which the engine reads as "hold whatever you already own".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Common.calendar import TradingCalendar, default_calendar

FREQUENCIES = ("daily", "weekly", "monthly", "quarterly", "on_signal_change")


def rebalance_dates(
    dates: pd.DatetimeIndex,
    frequency: str = "weekly",
    calendar: TradingCalendar | None = None,
) -> pd.DatetimeIndex:
    """Dates on which new targets are computed."""
    if frequency not in FREQUENCIES:
        raise ValueError(f"unknown rebalance frequency {frequency!r}")
    if frequency in {"daily", "on_signal_change"}:
        return pd.DatetimeIndex(dates)
    return (calendar or default_calendar()).rebalance_dates(
        pd.DatetimeIndex(dates), frequency
    )


def no_trade_band(
    targets: pd.DataFrame, epsilon: float = 0.0
) -> pd.DataFrame:
    """Carry the previous target forward for names that barely moved.

    Churning a position from 2.00% to 2.01% pays the spread twice and changes
    nothing, so below ``epsilon`` the old weight stands.
    """
    if epsilon <= 0.0:
        return targets

    values = targets.to_numpy(dtype=float, copy=True)
    previous = np.zeros(values.shape[1])
    for row in range(values.shape[0]):
        proposed = values[row]
        held = np.abs(proposed - previous) < epsilon
        proposed[held] = previous[held]
        values[row] = proposed
        previous = proposed
    return pd.DataFrame(values, index=targets.index, columns=targets.columns)


def limit_turnover(targets: pd.DataFrame, maximum: float | None) -> pd.DataFrame:
    """Cap one-way turnover per rebalance by moving only part of the way to target."""
    if maximum is None or maximum <= 0.0:
        return targets

    values = targets.to_numpy(dtype=float, copy=True)
    previous = np.zeros(values.shape[1])
    for row in range(values.shape[0]):
        proposed = values[row]
        traded = np.abs(proposed - previous).sum()
        if traded > maximum:
            proposed = previous + (proposed - previous) * (maximum / traded)
        values[row] = proposed
        previous = proposed
    return pd.DataFrame(values, index=targets.index, columns=targets.columns)


def drop_unchanged(targets: pd.DataFrame, tolerance: float = 1e-12) -> pd.DataFrame:
    """Blank rows identical to the previous instruction, so ``on_signal_change`` trades less."""
    changed = targets.diff().abs().sum(axis=1) > tolerance
    changed.iloc[0] = True
    return targets.where(changed, np.nan)


def expand(targets: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Place rebalance instructions on the full date index; other rows are NaN."""
    return targets.reindex(index=pd.DatetimeIndex(dates))
