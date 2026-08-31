"""The declarative description of a strategy, and how it is read from TOML.

The spec is data, not code: it hashes, it serialises into every result directory,
and two identical specs over identical data must produce identical outputs.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from Common.provenance import hash_payload
from Common.types import Partition
from Portfolio.constraints import ConstraintChain, build_chain
from Portfolio.constructors import get_constructor
from Signals.library import get_signal
from Strategy.BackTest.costs import CostModel
from Strategy.BackTest.engine import ExecutionConfig


@dataclass(frozen=True)
class UniverseSpec:
    """Which symbols the strategy may consider, before the tradability filter."""

    region: str = "US"
    asset_class: str = "stock"
    interval: str = "1d"
    symbols: tuple[str, ...] | None = None
    top_n_by_liquidity: int | None = 500
    limit: int | None = None

    @property
    def partition(self) -> Partition:
        return Partition(self.region, self.asset_class, self.interval)


@dataclass(frozen=True)
class ComponentSpec:
    """A registry name plus its keyword arguments."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict[str, Any] | str, default: str) -> ComponentSpec:
        if isinstance(payload, str):
            return cls(payload, {})
        data = dict(payload)
        return cls(str(data.pop("name", default)), data)


@dataclass(frozen=True)
class StrategySpec:
    """Everything needed to reproduce one backtest."""

    name: str
    universe: UniverseSpec = field(default_factory=UniverseSpec)
    signal: ComponentSpec = field(default_factory=lambda: ComponentSpec("momentum_12_1", {}))
    constructor: ComponentSpec = field(
        default_factory=lambda: ComponentSpec("quantile_long_short", {})
    )
    constraints: dict[str, Any] = field(default_factory=dict)
    costs: CostModel = field(default_factory=CostModel)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    start: str | None = None
    end: str | None = None
    rebalance: str = "weekly"
    epsilon: float = 0.0
    max_turnover: float | None = None
    engine: str = "vectorised"
    seed: int = 0

    # --- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StrategySpec:
        data = dict(payload)
        universe = UniverseSpec(**_tuple_symbols(data.pop("universe", {})))
        signal = ComponentSpec.parse(data.pop("signal", {}), "momentum_12_1")
        constructor = ComponentSpec.parse(data.pop("constructor", {}), "quantile_long_short")
        constraints = dict(data.pop("constraints", {}))
        costs = CostModel(**data.pop("costs", {}))

        execution_payload = dict(data.pop("execution", {}))
        for shared in ("execution_lag", "initial_capital"):
            if shared in data:
                execution_payload.setdefault(shared, data.pop(shared))
        execution = ExecutionConfig(**execution_payload)

        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown strategy fields: {sorted(unknown)}")

        return cls(
            universe=universe,
            signal=signal,
            constructor=constructor,
            constraints=constraints,
            costs=costs,
            execution=execution,
            **data,
        )

    @classmethod
    def load(cls, path: Path | str) -> StrategySpec:
        with Path(path).open("rb") as handle:
            return cls.from_dict(tomllib.load(handle))

    def with_overrides(self, **overrides: Any) -> StrategySpec:
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    # --- resolution --------------------------------------------------------
    def build_signal(self):
        return get_signal(self.signal.name, **self.signal.params)

    def build_constructor(self):
        return get_constructor(self.constructor.name, **self.constructor.params)

    def build_constraints(self) -> ConstraintChain:
        return build_chain(self.constraints)

    @property
    def start_ts(self) -> pd.Timestamp | None:
        return None if self.start is None else pd.Timestamp(self.start, tz="UTC")

    @property
    def end_ts(self) -> pd.Timestamp | None:
        return None if self.end is None else pd.Timestamp(self.end, tz="UTC")

    # --- identity ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["costs"] = self.costs.to_dict()
        return payload

    @property
    def hash(self) -> str:
        """Identity of the configuration, ignoring the window it is run over."""
        payload = self.to_dict()
        payload.pop("start", None)
        payload.pop("end", None)
        payload.pop("name", None)
        return hash_payload(payload)

    @property
    def full_hash(self) -> str:
        return hash_payload(self.to_dict())


def _tuple_symbols(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    symbols = data.get("symbols")
    if symbols is not None:
        data["symbols"] = tuple(str(item).upper() for item in symbols)
    return data
