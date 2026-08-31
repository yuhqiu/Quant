"""Panel-wide metrics: cross-sectional ranks and market-relative risk.

These run after the per-symbol matrices exist, because each value depends on
every other symbol on the same date.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Common.io import matrix_path, read_matrix, write_matrix
from MetricsGeneration.indicators import ANNUALIZE, TRADING_DAYS

RANK_SOURCES = ("ret_21d", "mom_12_1", "vol_20d", "advd_20", "rsi_14")
ZSCORE_SOURCES = ("ret_21d", "mom_12_1")

BETA_WINDOW = TRADING_DAYS
BETA_MIN_PERIODS = 126
MIN_CROSS_SECTION = 20

MARKET_FILENAME = "_market.parquet"


def _require_breadth(frame: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """Blank out dates where too few symbols trade for a comparison to mean anything."""
    enough = frame.count(axis=1) >= MIN_CROSS_SECTION
    return values.where(enough, axis=0)


def cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return _require_breadth(frame, frame.rank(axis=1, pct=True))


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    centered = frame.sub(frame.mean(axis=1), axis=0)
    scaled = centered.div(frame.std(axis=1).replace(0.0, np.nan), axis=0)
    return _require_breadth(frame, scaled)


def build_market_returns(daily_returns: pd.DataFrame) -> pd.Series:
    """Equal-weight universe return, used as the market factor until a real index exists."""
    breadth = daily_returns.count(axis=1)
    market = daily_returns.mean(axis=1).where(breadth >= MIN_CROSS_SECTION)
    market.name = "mkt_ret_1d"
    return market


def market_relative(
    daily_returns: pd.DataFrame,
    market_returns: pd.Series,
    chunk_size: int = 500,
) -> dict[str, pd.DataFrame]:
    """Beta, correlation and idiosyncratic volatility against the market proxy.

    Rolling covariance over a five-thousand-column panel would allocate several
    gigabytes at once, so symbols are processed in column blocks and cast down to
    float32 as each block finishes.
    """
    market_variance = market_returns.rolling(
        BETA_WINDOW, min_periods=BETA_MIN_PERIODS
    ).var()
    safe_variance = market_variance.replace(0.0, np.nan)

    blocks: dict[str, list[pd.DataFrame]] = {
        "beta_252d": [], "corr_mkt_252d": [], "idio_vol_252d": []
    }
    columns = list(daily_returns.columns)

    for start in range(0, len(columns), chunk_size):
        block = daily_returns[columns[start : start + chunk_size]]
        rolling = block.rolling(BETA_WINDOW, min_periods=BETA_MIN_PERIODS)

        correlation = rolling.corr(market_returns)
        beta = rolling.cov(market_returns).div(safe_variance, axis=0)
        idiosyncratic = rolling.std() * np.sqrt((1.0 - correlation**2).clip(lower=0.0))

        blocks["beta_252d"].append(beta.astype("float32"))
        blocks["corr_mkt_252d"].append(correlation.astype("float32"))
        blocks["idio_vol_252d"].append((idiosyncratic * ANNUALIZE).astype("float32"))

    return {name: pd.concat(parts, axis=1) for name, parts in blocks.items()}


def run(metrics_dir: Path, verbose: bool = True) -> list[str]:
    """Derive and write every panel-wide matrix. Returns the metric names written."""
    written: list[str] = []

    def emit(name: str, frame: pd.DataFrame) -> None:
        write_matrix(frame.astype("float32"), matrix_path(metrics_dir, name))
        written.append(name)
        if verbose:
            print(f"  cross-section: {name}")

    for source in RANK_SOURCES:
        path = matrix_path(metrics_dir, source)
        if not path.exists():
            continue
        frame = read_matrix(path)
        emit(f"cs_rank_{source}", cross_sectional_rank(frame))
        if source in ZSCORE_SOURCES:
            emit(f"cs_z_{source}", cross_sectional_zscore(frame))
        del frame

    daily_returns = read_matrix(matrix_path(metrics_dir, "ret_1d"))
    market_returns = build_market_returns(daily_returns)

    for name, frame in market_relative(daily_returns, market_returns).items():
        emit(name, frame)
    del daily_returns

    market_index = (1.0 + market_returns.fillna(0.0)).cumprod()
    market_frame = pd.DataFrame(
        {
            "mkt_ret_1d": market_returns,
            "mkt_ret_21d": market_index / market_index.shift(21) - 1.0,
            "mkt_ret_252d": market_index / market_index.shift(TRADING_DAYS) - 1.0,
            "mkt_index": market_index.where(market_returns.notna()),
            "mkt_vol_20d": market_returns.rolling(20).std() * ANNUALIZE,
        }
    )
    write_matrix(market_frame.astype("float64"), metrics_dir / MARKET_FILENAME)
    if verbose:
        print(f"  cross-section: {MARKET_FILENAME}")

    monthly_returns = read_matrix(matrix_path(metrics_dir, "ret_21d"))
    emit(
        "rel_ret_21d",
        monthly_returns.sub(market_frame["mkt_ret_21d"], axis=0),
    )

    return written
