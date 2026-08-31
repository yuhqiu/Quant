"""Strategy definitions and the backtest driver."""

from __future__ import annotations

from .BackTest import BacktestResult, CostModel, ExecutionConfig, MarketData, RiskRules
from .oos import record_touch, touches
from .runner import (
    baselines,
    build_panel,
    build_weights,
    buy_and_hold,
    run_spec,
    select_universe,
    sweep,
)
from .spec import ComponentSpec, StrategySpec, UniverseSpec
from .walkforward import Split, in_sample_split, walk_forward

__all__ = [
    "BacktestResult",
    "ComponentSpec",
    "CostModel",
    "ExecutionConfig",
    "MarketData",
    "RiskRules",
    "Split",
    "StrategySpec",
    "UniverseSpec",
    "baselines",
    "build_panel",
    "build_weights",
    "buy_and_hold",
    "in_sample_split",
    "record_touch",
    "run_spec",
    "select_universe",
    "sweep",
    "touches",
    "walk_forward",
]
