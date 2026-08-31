"""Constraints as a composable chain, applied in the order they are listed.

Each constraint takes weights and returns weights. Renormalisation happens inside
the constraint that needs it, so the chain stays order-independent in spirit and
explicit in practice.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .base import MarketContext, cap_and_refill, normalize_gross


@runtime_checkable
class Constraint(Protocol):
    def apply(self, weights: pd.DataFrame, context: MarketContext) -> pd.DataFrame: ...


@dataclass(frozen=True)
class MaxWeightPerName:
    """Cap single-name concentration, then put the freed weight back on the others."""

    limit: float = 0.05
    iterations: int = 12

    def apply(self, weights: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
        return cap_and_refill(weights, self.limit, iterations=self.iterations)


@dataclass(frozen=True)
class MaxGroupWeight:
    """Cap net exposure to any one sector or group."""

    limit: float = 0.25

    def apply(self, weights: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
        if context.groups is None:
            return weights
        labels = context.groups.reindex(weights.columns).dropna()
        result = weights.copy()
        for name in labels.unique():
            members = list(labels.index[labels == name])
            block = result[members]
            exposure = block.sum(axis=1)
            excess = exposure.abs() > self.limit
            if not excess.any():
                continue
            scale = (self.limit / exposure.abs().replace(0.0, np.nan)).clip(upper=1.0)
            result.loc[excess, members] = block.loc[excess].mul(scale[excess], axis=0)
        return result


@dataclass(frozen=True)
class MaxGross:
    limit: float = 1.0

    def apply(self, weights: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
        gross = weights.abs().sum(axis=1)
        scale = (self.limit / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
        return weights.mul(scale, axis=0)


@dataclass(frozen=True)
class MaxNet:
    """Bound directional exposure by shifting weight uniformly across held names."""

    limit: float = 0.1

    def apply(self, weights: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
        held = weights != 0.0
        count = held.sum(axis=1).replace(0, np.nan)
        net = weights.sum(axis=1)
        excess = net.clip(lower=-self.limit, upper=self.limit) - net
        return weights.add(held.div(count, axis=0).mul(excess, axis=0).fillna(0.0))


@dataclass(frozen=True)
class MaxBeta:
    """Bound portfolio beta by scaling the long and short books apart."""

    limit: float = 0.2

    def apply(self, weights: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
        if context.beta is None:
            return weights
        exposures = context.beta.reindex(index=weights.index, columns=weights.columns)
        portfolio_beta = (weights * exposures.fillna(1.0)).sum(axis=1)
        breach = portfolio_beta.abs() > self.limit
        if not breach.any():
            return weights

        held = weights != 0.0
        count = held.sum(axis=1).replace(0, np.nan)
        adjustment = portfolio_beta.clip(lower=-self.limit, upper=self.limit) - portfolio_beta
        mean_beta = exposures.where(held).mean(axis=1).replace(0.0, np.nan)
        delta = held.div(count, axis=0).mul((adjustment / mean_beta), axis=0)
        return weights.add(delta.fillna(0.0).where(breach, 0.0))


@dataclass(frozen=True)
class MinPositions:
    """Blank a date entirely rather than hold a portfolio too small to diversify."""

    minimum: int = 10

    def apply(self, weights: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
        enough = (weights != 0.0).sum(axis=1) >= self.minimum
        return weights.where(enough, 0.0)


@dataclass(frozen=True)
class RenormalizeGross:
    gross: float = 1.0

    def apply(self, weights: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
        return normalize_gross(weights, self.gross)


@dataclass(frozen=True)
class TradableOnly:
    def apply(self, weights: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
        return context.mask(weights).fillna(0.0)


@dataclass(frozen=True)
class ConstraintChain:
    constraints: tuple[Constraint, ...] = ()

    @classmethod
    def of(cls, constraints: Iterable[Constraint]) -> ConstraintChain:
        return cls(tuple(constraints))

    def apply(self, weights: pd.DataFrame, context: MarketContext) -> pd.DataFrame:
        result = weights
        for constraint in self.constraints:
            result = constraint.apply(result, context)
        return result.fillna(0.0)

    def __bool__(self) -> bool:
        return bool(self.constraints)


CONSTRAINT_REGISTRY: dict[str, type] = {
    "max_weight_per_name": MaxWeightPerName,
    "max_group_weight": MaxGroupWeight,
    "max_gross": MaxGross,
    "max_net": MaxNet,
    "max_beta": MaxBeta,
    "min_positions": MinPositions,
    "renormalize_gross": RenormalizeGross,
    "tradable_only": TradableOnly,
}


def build_chain(specification: dict[str, dict | float | int | None]) -> ConstraintChain:
    """``{"max_weight_per_name": 0.05, "min_positions": 20}`` -> a chain."""
    constraints: list[Constraint] = []
    for name, argument in specification.items():
        try:
            factory = CONSTRAINT_REGISTRY[name]
        except KeyError as error:
            raise ValueError(f"unknown constraint {name!r}") from error
        if argument is None:
            continue
        if isinstance(argument, dict):
            constraints.append(factory(**argument))
        else:
            field_name = next(iter(factory.__dataclass_fields__))
            constraints.append(factory(**{field_name: argument}))
    return ConstraintChain.of(constraints)
