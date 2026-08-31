"""The signal interface and the tradability filter every signal is masked by.

A signal is an opinion, not a position: it returns a score per symbol per date and
writes nothing to disk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from .panel import FeaturePanel
from .transforms import neutralize, winsorize


@runtime_checkable
class Signal(Protocol):
    name: str
    required_metrics: tuple[str, ...]

    def compute(self, panel: FeaturePanel) -> pd.DataFrame: ...


@dataclass(frozen=True)
class TradabilityFilter:
    """Names a strategy is allowed to hold at all, before any opinion is formed."""

    min_price: float = 5.0
    min_dollar_volume: float = 1_000_000.0
    max_stale_fraction: float = 0.5
    require_history: int = 252

    @property
    def required_metrics(self) -> tuple[str, ...]:
        return ("close", "advd_20", "stale_px_frac_20")

    def mask(self, panel: FeaturePanel) -> pd.DataFrame:
        """True where the symbol is tradable on that date."""
        close = panel.get("close")
        allowed = close.notna() & (close >= self.min_price)

        if self.min_dollar_volume > 0.0 and "advd_20" in panel:
            liquidity = panel.get("advd_20").reindex_like(close)
            allowed &= liquidity.notna() & (liquidity >= self.min_dollar_volume)

        if self.max_stale_fraction < 1.0 and "stale_px_frac_20" in panel:
            stale = panel.get("stale_px_frac_20").reindex_like(close)
            allowed &= stale.isna() | (stale <= self.max_stale_fraction)

        if self.require_history > 0:
            listed = close.notna().cumsum()
            allowed &= listed >= self.require_history

        return allowed


@dataclass
class BaseSignal(ABC):
    """Template: raw score, winsorise, neutralise, mask. Subclasses fill in the score."""

    name: str = "signal"
    required_metrics: tuple[str, ...] = ()
    neutralization: str = "zscore"
    winsor: tuple[float, float] | None = (0.01, 0.99)
    tradability: TradabilityFilter | None = field(default_factory=TradabilityFilter)

    @abstractmethod
    def raw(self, panel: FeaturePanel) -> pd.DataFrame:
        """The unprocessed opinion, higher meaning more attractive."""

    def compute(self, panel: FeaturePanel) -> pd.DataFrame:
        panel.require(self.metrics_needed())
        scores = self.raw(panel)

        if self.tradability is not None:
            scores = scores.where(self.tradability.mask(panel).reindex_like(scores))
        if self.winsor is not None:
            scores = winsorize(scores, *self.winsor)

        beta = panel.get("beta_252d") if self.neutralization == "beta" else None
        return neutralize(scores, self.neutralization, beta=beta)

    def metrics_needed(self) -> tuple[str, ...]:
        extra = self.tradability.required_metrics if self.tradability else ()
        if self.neutralization == "beta":
            extra += ("beta_252d",)
        return tuple(dict.fromkeys(self.required_metrics + extra))
