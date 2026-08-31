"""Reference signals. Each one is a few lines because the panel already did the work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .base import BaseSignal, Signal, TradabilityFilter
from .panel import FeaturePanel


@dataclass
class Momentum12_1(BaseSignal):
    """A year of return excluding the most recent month, the classic cross-sectional alpha."""

    name: str = "momentum_12_1"
    required_metrics: tuple[str, ...] = ("mom_12_1",)
    neutralization: str = "zscore"

    def raw(self, panel: FeaturePanel) -> pd.DataFrame:
        return panel.get("mom_12_1")


@dataclass
class ShortTermReversal(BaseSignal):
    """Last week's losers bounce: the sign is negative on purpose."""

    name: str = "reversal_5d"
    required_metrics: tuple[str, ...] = ("ret_5d",)
    neutralization: str = "zscore"

    def raw(self, panel: FeaturePanel) -> pd.DataFrame:
        return -panel.get("ret_5d")


@dataclass
class LowVolatility(BaseSignal):
    """Low realised volatility has historically earned more per unit of risk."""

    name: str = "low_vol_20d"
    required_metrics: tuple[str, ...] = ("vol_20d",)
    neutralization: str = "zscore"

    def raw(self, panel: FeaturePanel) -> pd.DataFrame:
        return -panel.get("vol_20d")


@dataclass
class MetricSignal(BaseSignal):
    """Use any single metric as a signal, optionally inverted. Handy for probes."""

    name: str = "metric"
    metric: str = "mom_12_1"
    sign: float = 1.0
    required_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.required_metrics = (self.metric,)
        if self.name == "metric":
            self.name = f"{'neg_' if self.sign < 0 else ''}{self.metric}"

    def raw(self, panel: FeaturePanel) -> pd.DataFrame:
        return panel.get(self.metric) * self.sign


@dataclass
class RandomSignal(BaseSignal):
    """Seeded noise. The null hypothesis every real signal must beat."""

    name: str = "random"
    required_metrics: tuple[str, ...] = ("close",)
    neutralization: str = "zscore"
    seed: int = 0

    def raw(self, panel: FeaturePanel) -> pd.DataFrame:
        import numpy as np

        close = panel.get("close")
        generator = np.random.default_rng(self.seed)
        noise = generator.standard_normal(close.shape)
        return pd.DataFrame(noise, index=close.index, columns=close.columns).where(
            close.notna()
        )


REGISTRY: dict[str, type] = {
    "momentum_12_1": Momentum12_1,
    "reversal_5d": ShortTermReversal,
    "low_vol_20d": LowVolatility,
    "metric": MetricSignal,
    "random": RandomSignal,
}


def signal_names() -> list[str]:
    return sorted(REGISTRY)


def get_signal(name: str, **params: Any) -> Signal:
    try:
        factory = REGISTRY[name]
    except KeyError as error:
        raise ValueError(
            f"unknown signal {name!r}; known: {', '.join(signal_names())}"
        ) from error

    tradability = params.pop("tradability", None)
    if isinstance(tradability, dict):
        params["tradability"] = TradabilityFilter(**tradability)
    elif tradability is not None:
        params["tradability"] = tradability
    return factory(**params)
