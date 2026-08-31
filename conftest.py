"""Shared pytest fixtures. Synthetic, deterministic, and small enough to run in seconds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


def make_bars(
    periods: int = 400,
    seed: int = 0,
    start: str = "2020-01-01",
    drift: float = 0.0003,
    volatility: float = 0.02,
    dividends: bool = False,
) -> pd.DataFrame:
    """One symbol of plausible OHLCV with adjustment columns."""
    generator = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=periods, tz="UTC", name="date")

    steps = generator.normal(drift, volatility, periods)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = close * (1.0 + generator.normal(0.0, 0.003, periods))
    high = np.maximum(open_, close) * (1.0 + np.abs(generator.normal(0.0, 0.004, periods)))
    low = np.minimum(open_, close) * (1.0 - np.abs(generator.normal(0.0, 0.004, periods)))
    volume = generator.integers(200_000, 2_000_000, periods).astype(float)

    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "adj_close": close,
            "dividend": 0.0,
            "split_ratio": 0.0,
            "repaired": 0.0,
        },
        index=index,
    )
    if dividends:
        frame.iloc[periods // 3, frame.columns.get_loc("dividend")] = 0.5
    return frame


def make_panel_frames(
    symbols: tuple[str, ...] = tuple(f"SYM{i:02d}" for i in range(30)),
    periods: int = 400,
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    """Wide date x symbol frames covering everything the engines and signals read."""
    from MetricsGeneration.indicators import compute_labels, compute_metrics

    per_symbol = {
        symbol: make_bars(periods, seed=seed + offset)
        for offset, symbol in enumerate(symbols)
    }
    metrics = {symbol: compute_metrics(bars) for symbol, bars in per_symbol.items()}
    labels = {symbol: compute_labels(bars) for symbol, bars in per_symbol.items()}

    combined = pd.concat(metrics, axis=1)
    frames = {
        name: combined.xs(name, axis=1, level=1).astype("float64")
        for name in next(iter(metrics.values())).columns
    }
    label_block = pd.concat(labels, axis=1)
    for name in next(iter(labels.values())).columns:
        frames[name] = label_block.xs(name, axis=1, level=1).astype("float64")
    return frames


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture(scope="session")
def panel_frames() -> dict[str, pd.DataFrame]:
    return make_panel_frames()


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Point every project path at a throwaway directory."""
    from Common import config

    monkeypatch.setenv("QUANT_DATA_ROOT", str(tmp_path / "DataSource"))
    monkeypatch.setenv("QUANT_METRICS_ROOT", str(tmp_path / "Metrics"))
    monkeypatch.setenv("QUANT_RESULTS_ROOT", str(tmp_path / "Results"))
    config.reset_settings()
    yield config.settings()
    config.reset_settings()
