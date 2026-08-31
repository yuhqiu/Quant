"""Composition primitives: build one opinion out of several."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Signal
from .panel import FeaturePanel
from .transforms import rank_normalize, zscore


@dataclass
class CompositeSignal:
    """Weighted blend of component signals after a common normalisation."""

    components: list[Signal]
    weights: list[float] | None = None
    name: str = "composite"
    method: str = "zscore"

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("a composite needs at least one component")
        if self.weights is None:
            self.weights = [1.0 / len(self.components)] * len(self.components)
        if len(self.weights) != len(self.components):
            raise ValueError("weights and components must be the same length")
        total = sum(abs(weight) for weight in self.weights)
        if total == 0.0:
            raise ValueError("weights cannot all be zero")
        self.weights = [weight / total for weight in self.weights]

    @property
    def required_metrics(self) -> tuple[str, ...]:
        names: tuple[str, ...] = ()
        for component in self.components:
            names += tuple(component.required_metrics)
        return tuple(dict.fromkeys(names))

    def _normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.method == "rank":
            return rank_normalize(frame)
        if self.method == "zscore":
            return zscore(frame)
        if self.method == "raw":
            return frame
        raise ValueError(f"unknown combine method {self.method!r}")

    def compute(self, panel: FeaturePanel) -> pd.DataFrame:
        blended: pd.DataFrame | None = None
        for component, weight in zip(self.components, self.weights, strict=True):
            piece = self._normalize(component.compute(panel)) * weight
            blended = piece if blended is None else blended.add(piece, fill_value=0.0)
        assert blended is not None
        return blended


@dataclass
class WeightedSignal(CompositeSignal):
    method: str = "raw"


@dataclass
class RankCombine(CompositeSignal):
    method: str = "rank"


@dataclass
class ZScoreCombine(CompositeSignal):
    method: str = "zscore"
