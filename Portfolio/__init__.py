"""Positions from signals: constructors, constraints and rebalance scheduling."""

from __future__ import annotations

import pandas as pd

from .base import MarketContext, PortfolioConstructor, normalize_gross
from .constraints import (
    CONSTRAINT_REGISTRY,
    Constraint,
    ConstraintChain,
    MaxBeta,
    MaxGross,
    MaxGroupWeight,
    MaxNet,
    MaxWeightPerName,
    MinPositions,
    RenormalizeGross,
    TradableOnly,
    build_chain,
)
from .constructors import (
    REGISTRY,
    InverseVolWeighted,
    MeanVariance,
    QuantileLongShort,
    ScoreProportional,
    TopNEqualWeight,
    constructor_names,
    get_constructor,
    ledoit_wolf,
)
from .schedule import (
    FREQUENCIES,
    drop_unchanged,
    expand,
    limit_turnover,
    no_trade_band,
    rebalance_dates,
)

__all__ = [
    "CONSTRAINT_REGISTRY",
    "Constraint",
    "ConstraintChain",
    "FREQUENCIES",
    "InverseVolWeighted",
    "MarketContext",
    "MaxBeta",
    "MaxGross",
    "MaxGroupWeight",
    "MaxNet",
    "MaxWeightPerName",
    "MeanVariance",
    "MinPositions",
    "PortfolioConstructor",
    "QuantileLongShort",
    "REGISTRY",
    "RenormalizeGross",
    "ScoreProportional",
    "TopNEqualWeight",
    "TradableOnly",
    "build_chain",
    "build_targets",
    "constructor_names",
    "drop_unchanged",
    "expand",
    "get_constructor",
    "ledoit_wolf",
    "limit_turnover",
    "no_trade_band",
    "normalize_gross",
    "rebalance_dates",
]


def build_targets(
    scores: pd.DataFrame,
    context: MarketContext,
    constructor: PortfolioConstructor,
    constraints: ConstraintChain | None = None,
    rebalance: str = "weekly",
    epsilon: float = 0.0,
    max_turnover: float | None = None,
    calendar=None,
) -> pd.DataFrame:
    """Score -> target weights on rebalance dates, NaN elsewhere meaning "hold".

    The no-trade band and the turnover cap are applied against the previous
    *target*, not the previous drifted holding, which is the standard vectorised
    approximation and is stated here so nobody mistakes it for exact.
    """
    dates = rebalance_dates(scores.index, rebalance, calendar)
    at_rebalance = scores.reindex(index=dates)

    weights = constructor.target_weights(at_rebalance, context.slice(dates))
    if constraints:
        weights = constraints.apply(weights, context.slice(dates))

    weights = no_trade_band(weights.fillna(0.0), epsilon)
    weights = limit_turnover(weights, max_turnover)

    if rebalance == "on_signal_change":
        weights = drop_unchanged(weights)

    return expand(weights, scores.index)
