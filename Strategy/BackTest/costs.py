"""Trading frictions. Costs are always on.

A cost-free backtest is a marketing document, not a result, so switching them off
requires an explicit ``enabled=False`` rather than an omission.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

BPS = 1e-4
TRADING_DAYS = 252.0


@dataclass(frozen=True)
class CostModel:
    """Commission, spread, market impact, borrow and cash interest."""

    commission_bps: float = 0.5
    commission_per_share: float = 0.0
    min_commission: float = 0.0
    half_spread_bps: float = 2.0
    estimate_spread_from_range: bool = True
    range_spread_factor: float = 0.02
    max_half_spread: float = 0.01
    impact_coefficient: float = 0.1
    borrow_bps_annual: float = 50.0
    cash_rate_annual: float = 0.0
    enabled: bool = True

    def commission(self, notional: np.ndarray, shares: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return np.zeros_like(notional)
        charge = notional * self.commission_bps * BPS
        charge += np.abs(shares) * self.commission_per_share
        traded = notional > 0.0
        if self.min_commission > 0.0:
            charge = np.where(traded, np.maximum(charge, self.min_commission), 0.0)
        return np.where(traded, charge, 0.0)

    def spread(
        self,
        notional: np.ndarray,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        close: np.ndarray | None = None,
    ) -> np.ndarray:
        """Half the quoted spread, estimated from the bar range when unknown."""
        if not self.enabled:
            return np.zeros_like(notional)

        half = np.full(notional.shape, self.half_spread_bps * BPS)
        if self.estimate_spread_from_range and high is not None and low is not None and close is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                # A small fraction of the day's range: wide-ranging names quote wider,
                # but the range itself is an order of magnitude larger than the spread.
                estimated = self.range_spread_factor * (high - low) / np.where(close > 0.0, close, np.nan)
            estimated = np.nan_to_num(estimated, nan=0.0, posinf=0.0, neginf=0.0)
            half = np.maximum(half, np.clip(estimated, 0.0, self.max_half_spread))
        return notional * half

    def slippage(
        self, notional: np.ndarray, sigma: np.ndarray, advd: np.ndarray
    ) -> np.ndarray:
        """Square-root impact: ``k * sigma * sqrt(notional / ADV$)`` of notional."""
        if not self.enabled or self.impact_coefficient == 0.0:
            return np.zeros_like(notional)

        daily_sigma = np.nan_to_num(sigma, nan=0.02) / np.sqrt(TRADING_DAYS)
        capacity = np.where(advd > 0.0, advd, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            participation = np.sqrt(np.clip(notional / capacity, 0.0, 1.0))
        participation = np.nan_to_num(participation, nan=0.0)
        return notional * self.impact_coefficient * daily_sigma * participation

    def borrow(self, short_value: float, days: float = 1.0) -> float:
        if not self.enabled or short_value <= 0.0:
            return 0.0
        return short_value * self.borrow_bps_annual * BPS * days / TRADING_DAYS

    def interest(self, cash: float, days: float = 1.0) -> float:
        """Interest on idle cash; negative cash pays the same rate."""
        if not self.enabled or self.cash_rate_annual == 0.0:
            return 0.0
        return cash * self.cash_rate_annual * days / TRADING_DAYS

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


ZERO_COST = CostModel(enabled=False)
