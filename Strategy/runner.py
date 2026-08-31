"""Bind a :class:`StrategySpec` to data and run it.

This is the only place that knows the order of the pipeline: universe -> panel ->
signal -> weights -> simulation -> artifacts.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Common.io import matrix_path, read_matrix
from Common.logging import get_logger
from Common.provenance import manifest_hashes
from Common.types import Partition
from MetricsGeneration import MANIFEST_FILENAME
from Portfolio import build_targets
from Portfolio.base import MarketContext
from Signals.base import BaseSignal
from Signals.panel import FeaturePanel
from Strategy.BackTest import BacktestResult, MarketData
from Strategy.BackTest import run as run_engine
from Strategy.BackTest.result import make_run_id
from Strategy.spec import StrategySpec

log = get_logger(__name__)


def select_universe(
    partition: Partition,
    top_n: int | None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    symbols: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[str]:
    """The most liquid ``top_n`` names over the window, measured by median ADV$."""
    liquidity_path = matrix_path(partition.metrics_dir, "advd_20")
    if symbols is not None:
        names = list(symbols)
    elif top_n is None or not liquidity_path.exists():
        names = list(read_matrix(matrix_path(partition.metrics_dir, "close")).columns)
    else:
        liquidity = read_matrix(liquidity_path)
        if start is not None:
            liquidity = liquidity.loc[liquidity.index >= start]
        if end is not None:
            liquidity = liquidity.loc[liquidity.index <= end]
        ranked = liquidity.median(axis=0, skipna=True).dropna().sort_values(ascending=False)
        names = list(ranked.head(top_n).index)
        del liquidity

    names = sorted(names)
    return names[:limit] if limit else names


def build_panel(spec: StrategySpec) -> FeaturePanel:
    partition = spec.universe.partition
    symbols = select_universe(
        partition,
        spec.universe.top_n_by_liquidity,
        spec.start_ts,
        spec.end_ts,
        spec.universe.symbols,
        spec.universe.limit,
    )
    if not symbols:
        raise ValueError(f"no symbols selected for {partition}")
    # Full history is loaded on purpose: a tradability rule that asks for a year of
    # listing must be able to see the year before the backtest starts.
    return FeaturePanel(partition=partition, symbols=tuple(symbols))


def build_weights(spec: StrategySpec, panel: FeaturePanel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (scores, target weights). Weights are NaN on non-rebalance dates."""
    signal = spec.build_signal()
    scores = signal.compute(panel)

    context = MarketContext.from_panel(panel)
    if isinstance(signal, BaseSignal) and signal.tradability is not None:
        context.tradable = signal.tradability.mask(panel)

    targets = build_targets(
        scores=scores,
        context=context,
        constructor=spec.build_constructor(),
        constraints=spec.build_constraints(),
        rebalance=spec.rebalance,
        epsilon=spec.epsilon,
        max_turnover=spec.max_turnover,
    )
    return scores, targets


def run_spec(
    spec: StrategySpec,
    engine: str | None = None,
    save: bool = True,
    root: Path | str | None = None,
    run_id: str | None = None,
    **engine_kwargs: Any,
) -> BacktestResult:
    """Run one strategy end to end and return its result."""
    np.random.seed(spec.seed)

    panel = build_panel(spec)
    scores, targets = build_weights(spec, panel)

    data = MarketData.from_panel(panel, panel.universe, spec.start_ts, spec.end_ts)
    if data.shape[0] == 0:
        raise ValueError("the requested window contains no bars")

    payload = spec.to_dict()
    payload["spec_hash"] = spec.hash
    payload["full_hash"] = spec.full_hash
    payload["symbols"] = len(panel.universe)
    payload["inputs"] = manifest_hashes(
        [spec.universe.partition.metrics_dir / MANIFEST_FILENAME]
    )

    result = run_engine(
        targets=targets,
        data=data,
        costs=spec.costs,
        config=spec.execution,
        engine=engine or spec.engine,
        name=spec.name,
        run_id=run_id or make_run_id(spec.full_hash),
        spec=payload,
        **engine_kwargs,
    )

    problems = result.check_invariants()
    if problems:
        log.error("backtest invariants violated", extra={"problems": problems})
        result.metrics["invariant_failures"] = problems

    if save:
        target = result.save(root)
        log.info("backtest saved", extra={"path": str(target), "run": result.run_id})
    return result


def sweep(
    spec: StrategySpec,
    grid: dict[str, Iterable[Any]],
    section: str = "constructor",
    save: bool = False,
) -> pd.DataFrame:
    """Run the whole parameter surface and report all of it, not the maximum.

    A strategy that works at exactly one parameter value is noise, so the caller
    gets every cell and has to look at the shape.
    """
    keys = list(grid)
    rows: list[dict[str, Any]] = []

    for combination in itertools.product(*(list(grid[key]) for key in keys)):
        params = dict(zip(keys, combination, strict=True))
        candidate = _with_params(spec, section, params)
        result = run_spec(candidate, save=save)
        rows.append({**params, **result.metrics, "spec_hash": candidate.hash})

    frame = pd.DataFrame(rows)
    frame.attrs["configurations"] = len(rows)
    return frame


def _with_params(spec: StrategySpec, section: str, params: dict[str, Any]) -> StrategySpec:
    if section == "constructor":
        component = replace(spec.constructor, params={**spec.constructor.params, **params})
        return replace(spec, constructor=component)
    if section == "signal":
        component = replace(spec.signal, params={**spec.signal.params, **params})
        return replace(spec, signal=component)
    if section == "strategy":
        return spec.with_overrides(**params)
    raise ValueError(f"unknown sweep section {section!r}")


def baselines(spec: StrategySpec, save: bool = False) -> dict[str, BacktestResult]:
    """Buy-and-hold, equal-weight universe and a turnover-matched random signal."""
    results: dict[str, BacktestResult] = {}

    equal_weight = replace(
        spec,
        name=f"{spec.name}__equal_weight",
        signal=replace(spec.signal, name="random", params={"seed": spec.seed}),
        constructor=replace(
            spec.constructor, name="top_n_equal_weight", params={"n": 200, "long_only": True}
        ),
        rebalance="monthly",
    )
    results["equal_weight"] = run_spec(equal_weight, save=save)

    random_null = replace(
        spec,
        name=f"{spec.name}__random",
        signal=replace(spec.signal, name="random", params={"seed": spec.seed + 1}),
    )
    results["random"] = run_spec(random_null, save=save)
    return results


def buy_and_hold(
    symbol: str = "SPY",
    spec: StrategySpec | None = None,
    asset_class: str = "etf",
    save: bool = False,
) -> BacktestResult:
    """The benchmark every strategy has to beat before it is interesting."""
    spec = spec or StrategySpec(name="buy_and_hold")
    partition = Partition(spec.universe.region, asset_class, spec.universe.interval)
    panel = FeaturePanel(partition=partition, symbols=(symbol,))

    data = MarketData.from_panel(panel, (symbol,), spec.start_ts, spec.end_ts)
    targets = pd.DataFrame(np.nan, index=data.dates, columns=[symbol])
    # One instruction, then hold: that is what buy-and-hold means.
    targets.iloc[0] = 1.0

    result = run_engine(
        targets=targets,
        data=data,
        costs=spec.costs,
        config=spec.execution,
        engine="vectorised",
        name=f"buy_and_hold_{symbol}",
        spec={"name": f"buy_and_hold_{symbol}", "symbol": symbol},
    )
    if save:
        result.save()
    return result
