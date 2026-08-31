"""What a portfolio constructor is allowed to know, and the interface it implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass
class MarketContext:
    """Everything observable at decision time. No forward-looking columns allowed."""

    prices: pd.DataFrame
    tradable: pd.DataFrame | None = None
    volatility: pd.DataFrame | None = None
    beta: pd.DataFrame | None = None
    returns: pd.DataFrame | None = None
    groups: pd.Series | None = None
    rebalance_dates: pd.DatetimeIndex | None = None

    def mask(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.tradable is None:
            return frame
        allowed = self.tradable.reindex(index=frame.index, columns=frame.columns)
        return frame.where(allowed.fillna(False))

    def slice(self, dates: pd.DatetimeIndex) -> MarketContext:
        def cut(frame: pd.DataFrame | None) -> pd.DataFrame | None:
            return None if frame is None else frame.reindex(index=dates)

        return MarketContext(
            prices=self.prices.reindex(index=dates),
            tradable=cut(self.tradable),
            volatility=cut(self.volatility),
            beta=cut(self.beta),
            returns=self.returns,
            groups=self.groups,
            rebalance_dates=dates,
        )

    @classmethod
    def from_panel(cls, panel, rebalance_dates: pd.DatetimeIndex | None = None) -> MarketContext:
        """Assemble a context from a :class:`Signals.FeaturePanel`."""
        prices = panel.get("adj_close") if "adj_close" in panel else panel.get("close")
        return cls(
            prices=prices,
            volatility=panel.get("vol_20d") if "vol_20d" in panel else None,
            beta=panel.get("beta_252d") if "beta_252d" in panel else None,
            returns=panel.get("ret_1d") if "ret_1d" in panel else None,
            rebalance_dates=rebalance_dates,
        )


@runtime_checkable
class PortfolioConstructor(Protocol):
    name: str

    def target_weights(
        self, scores: pd.DataFrame, context: MarketContext
    ) -> pd.DataFrame: ...


def normalize_gross(weights: pd.DataFrame, gross: float) -> pd.DataFrame:
    """Scale each date so the sum of absolute weights equals ``gross``."""
    total = weights.abs().sum(axis=1).replace(0.0, float("nan"))
    return weights.div(total, axis=0).mul(gross).fillna(0.0)


def cap_and_refill(
    weights: pd.DataFrame, cap: float, gross: float | None = None, iterations: int = 12
) -> pd.DataFrame:
    """Clip to ``cap`` and push the freed weight onto names that still have room.

    Clipping alone shrinks the book and renormalising alone breaks the cap, so the
    two are alternated until they agree.
    """
    import numpy as np

    target = weights.abs().sum(axis=1) if gross is None else pd.Series(gross, index=weights.index)
    result = weights
    for _ in range(iterations):
        capped = result.clip(lower=-cap, upper=cap)
        deficit = target - capped.abs().sum(axis=1)
        if float(deficit.abs().max()) < 1e-12:
            return capped
        room = (cap - capped.abs()).where(capped != 0.0, 0.0)
        share = room.div(room.sum(axis=1).replace(0.0, np.nan), axis=0)
        result = capped + np.sign(capped) * share.mul(deficit, axis=0).fillna(0.0)
    return result.clip(lower=-cap, upper=cap)
