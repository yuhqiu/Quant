"""Concrete ways to turn a score into target weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .base import MarketContext, cap_and_refill, normalize_gross


def _ranked(scores: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
    return context.mask(scores).rank(axis=1, pct=True)


@dataclass(frozen=True)
class TopNEqualWeight:
    """Hold the best ``n`` names equally. The simplest thing that could work."""

    n: int = 50
    long_only: bool = True
    gross: float = 1.0
    name: str = "top_n_equal_weight"

    def target_weights(self, scores: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
        masked = context.mask(scores)
        descending = masked.rank(axis=1, ascending=False, method="first")
        longs = (descending <= self.n).astype("float64")

        if self.long_only:
            return normalize_gross(longs, self.gross)

        ascending = masked.rank(axis=1, ascending=True, method="first")
        shorts = (ascending <= self.n).astype("float64")
        return normalize_gross(longs - shorts, self.gross)


@dataclass(frozen=True)
class QuantileLongShort:
    """Long the top quantile, short the bottom, sized to a fixed gross and net."""

    quantiles: int = 5
    gross: float = 1.0
    net: float = 0.0
    name: str = "quantile_long_short"

    def target_weights(self, scores: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
        ranked = _ranked(scores, context)
        edge = 1.0 / self.quantiles

        longs = (ranked > 1.0 - edge).astype("float64")
        shorts = (ranked <= edge).astype("float64")

        long_book = longs.div(longs.sum(axis=1).replace(0.0, np.nan), axis=0)
        short_book = shorts.div(shorts.sum(axis=1).replace(0.0, np.nan), axis=0)

        # gross = long + short, net = long - short: solve for each side's size.
        long_size = (self.gross + self.net) / 2.0
        short_size = (self.gross - self.net) / 2.0
        both = long_book.notna().any(axis=1) & short_book.notna().any(axis=1)

        weights = long_book.fillna(0.0) * long_size - short_book.fillna(0.0) * short_size
        return weights.where(both, 0.0)


@dataclass(frozen=True)
class ScoreProportional:
    """Weight by the score itself, capped per name so one outlier cannot take over."""

    cap_per_name: float = 0.05
    gross: float = 1.0
    demean: bool = True
    name: str = "score_proportional"

    def target_weights(self, scores: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
        masked = context.mask(scores)
        if self.demean:
            masked = masked.sub(masked.mean(axis=1), axis=0)
        weights = normalize_gross(masked.fillna(0.0), self.gross)
        return cap_and_refill(weights, self.cap_per_name, self.gross)


@dataclass(frozen=True)
class InverseVolWeighted:
    """Equal risk rather than equal money: size positions by 1 / volatility."""

    quantiles: int = 5
    gross: float = 1.0
    net: float = 0.0
    vol_metric: str = "vol_20d"
    floor: float = 0.05
    name: str = "inverse_vol_weighted"

    def target_weights(self, scores: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
        base = QuantileLongShort(self.quantiles, self.gross, self.net).target_weights(
            scores, context
        )
        if context.volatility is None:
            return base

        volatility = context.volatility.reindex(
            index=base.index, columns=base.columns
        ).clip(lower=self.floor)
        tilted = base.div(volatility).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        longs = normalize_gross(tilted.clip(lower=0.0), (self.gross + self.net) / 2.0)
        shorts = normalize_gross(tilted.clip(upper=0.0), (self.gross - self.net) / 2.0)
        return longs + shorts


@dataclass(frozen=True)
class MeanVariance:
    """Markowitz on the selected names, with Ledoit-Wolf shrinkage on the covariance.

    Sample covariance over thousands of symbols is mostly noise, so the estimate is
    shrunk toward a scaled identity before it is ever inverted.
    """

    risk_aversion: float = 5.0
    lookback: int = 252
    max_names: int = 100
    gross: float = 1.0
    long_only: bool = False
    name: str = "mean_variance"

    def target_weights(self, scores: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
        if context.returns is None:
            raise ValueError("MeanVariance needs context.returns")

        returns = context.returns
        weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        masked = context.mask(scores)

        for date in scores.index:
            row = masked.loc[date].dropna()
            if row.empty:
                continue
            selected = row.abs().nlargest(self.max_names).index
            window = returns.loc[returns.index <= date, selected].tail(self.lookback)
            window = window.dropna(axis=1, thresh=int(0.8 * len(window)))
            if window.shape[0] < 60 or window.shape[1] < 2:
                continue

            covariance = ledoit_wolf(window.fillna(0.0).to_numpy())
            expected = row[window.columns].to_numpy()
            try:
                raw = np.linalg.solve(self.risk_aversion * covariance, expected)
            except np.linalg.LinAlgError:
                continue
            if self.long_only:
                raw = np.clip(raw, 0.0, None)
            total = np.abs(raw).sum()
            if total > 0.0:
                weights.loc[date, window.columns] = raw / total * self.gross
        return weights


def ledoit_wolf(returns: np.ndarray) -> np.ndarray:
    """Sample covariance shrunk toward a scaled identity, shrinkage chosen analytically."""
    observations, variables = returns.shape
    centered = returns - returns.mean(axis=0, keepdims=True)
    sample = centered.T @ centered / observations

    mean_variance = np.trace(sample) / variables
    target = mean_variance * np.eye(variables)

    dispersion = np.linalg.norm(sample - target, "fro") ** 2
    squared = np.einsum("ij,ik->jk", centered**2, centered**2) / observations
    noise = float(np.sum(squared - sample**2)) / observations

    intensity = 0.0 if dispersion <= 0.0 else min(max(noise / dispersion, 0.0), 1.0)
    return intensity * target + (1.0 - intensity) * sample


REGISTRY: dict[str, type] = {
    "top_n_equal_weight": TopNEqualWeight,
    "quantile_long_short": QuantileLongShort,
    "score_proportional": ScoreProportional,
    "inverse_vol_weighted": InverseVolWeighted,
    "mean_variance": MeanVariance,
}


def constructor_names() -> list[str]:
    return sorted(REGISTRY)


def get_constructor(name: str, **params: Any):
    try:
        return REGISTRY[name](**params)
    except KeyError as error:
        raise ValueError(
            f"unknown constructor {name!r}; known: {', '.join(constructor_names())}"
        ) from error
