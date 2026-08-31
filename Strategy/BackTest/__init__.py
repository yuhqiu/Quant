"""Backtest engines: vectorised for scale, event-driven for path-dependent logic."""

from __future__ import annotations

import pandas as pd

from . import event, vectorised
from .costs import ZERO_COST, CostModel
from .engine import ExecutionConfig, MarketData, TradeLog, lagged_targets
from .event import RiskRules
from .result import (
    BacktestResult,
    build_equity_frame,
    latest_run,
    make_run_id,
)

ENGINES = ("vectorised", "event_driven")

__all__ = [
    "BacktestResult",
    "CostModel",
    "ENGINES",
    "ExecutionConfig",
    "MarketData",
    "RiskRules",
    "TradeLog",
    "ZERO_COST",
    "build_equity_frame",
    "event",
    "lagged_targets",
    "latest_run",
    "make_run_id",
    "run",
    "vectorised",
]


def run(
    targets: pd.DataFrame,
    data: MarketData,
    costs: CostModel | None = None,
    config: ExecutionConfig | None = None,
    engine: str = "vectorised",
    **kwargs,
) -> BacktestResult:
    """Dispatch to an engine. Both produce the same :class:`BacktestResult`."""
    if engine == "vectorised":
        return vectorised.run(targets, data, costs, config, **kwargs)
    if engine == "event_driven":
        return event.run(targets, data, costs, config, **kwargs)
    raise ValueError(f"unknown engine {engine!r}; known: {', '.join(ENGINES)}")
