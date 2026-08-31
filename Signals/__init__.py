"""Alpha signals: feature panel -> score per symbol per date."""

from __future__ import annotations

from .base import BaseSignal, Signal, TradabilityFilter
from .combine import CompositeSignal, RankCombine, WeightedSignal, ZScoreCombine
from .library import (
    REGISTRY,
    LowVolatility,
    MetricSignal,
    Momentum12_1,
    RandomSignal,
    ShortTermReversal,
    get_signal,
    signal_names,
)
from .panel import FeaturePanel
from .report import (
    SignalReport,
    autocorrelation,
    coverage,
    evaluate,
    information_coefficient,
    quantile_spread,
    save_report,
    turnover,
)
from .transforms import (
    beta_neutral,
    demean,
    group_neutral,
    neutralize,
    rank_normalize,
    winsorize,
    zscore,
)

__all__ = [
    "BaseSignal",
    "CompositeSignal",
    "FeaturePanel",
    "LowVolatility",
    "MetricSignal",
    "Momentum12_1",
    "REGISTRY",
    "RandomSignal",
    "RankCombine",
    "ShortTermReversal",
    "Signal",
    "SignalReport",
    "TradabilityFilter",
    "WeightedSignal",
    "ZScoreCombine",
    "autocorrelation",
    "beta_neutral",
    "coverage",
    "demean",
    "evaluate",
    "get_signal",
    "group_neutral",
    "information_coefficient",
    "neutralize",
    "quantile_spread",
    "rank_normalize",
    "save_report",
    "signal_names",
    "turnover",
    "winsorize",
    "zscore",
]
