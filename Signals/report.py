"""Signal quality: information coefficient, quantile spread, turnover, decay.

A signal must pass through here before it is allowed near a backtest, because a
score with no IC and 200% turnover is dead on arrival after costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from Common.config import settings
from Common.io import write_json, write_parquet
from Common.provenance import stamp
from Common.types import TRADING_DAYS

from .base import Signal
from .panel import FeaturePanel

DEFAULT_HORIZONS = (1, 5, 21)
MIN_NAMES = 20


def _row_corr(left: pd.DataFrame, right: pd.DataFrame, minimum: int) -> pd.Series:
    """Pearson correlation computed row by row over the symbols present in both."""
    valid = left.notna() & right.notna()
    a = left.where(valid)
    b = right.where(valid)

    a = a.sub(a.mean(axis=1), axis=0)
    b = b.sub(b.mean(axis=1), axis=0)

    covariance = (a * b).sum(axis=1)
    scale = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    correlation = covariance / scale.replace(0.0, np.nan)
    return correlation.where(valid.sum(axis=1) >= minimum)


def _row_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True)


def information_coefficient(
    scores: pd.DataFrame, forward_returns: pd.DataFrame, minimum: int = MIN_NAMES
) -> pd.Series:
    """Spearman rank IC per date: rank the scores, rank the outcomes, correlate."""
    aligned = forward_returns.reindex(index=scores.index, columns=scores.columns)
    return _row_corr(_row_rank(scores), _row_rank(aligned), minimum)


def quantile_spread(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    quantiles: int = 10,
    minimum: int = MIN_NAMES,
) -> pd.Series:
    """Mean forward return of the top bucket minus the bottom bucket, per date."""
    aligned = forward_returns.reindex(index=scores.index, columns=scores.columns)
    valid = scores.notna() & aligned.notna()
    ranked = scores.where(valid).rank(axis=1, pct=True)

    edge = 1.0 / quantiles
    top = aligned.where(valid & (ranked > 1.0 - edge)).mean(axis=1)
    bottom = aligned.where(valid & (ranked <= edge)).mean(axis=1)
    return (top - bottom).where(valid.sum(axis=1) >= minimum)


def quantile_returns(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    quantiles: int = 10,
) -> pd.DataFrame:
    """Mean forward return by score bucket, averaged over all dates."""
    aligned = forward_returns.reindex(index=scores.index, columns=scores.columns)
    valid = scores.notna() & aligned.notna()
    ranked = scores.where(valid).rank(axis=1, pct=True)

    rows = {}
    for bucket in range(quantiles):
        low, high = bucket / quantiles, (bucket + 1) / quantiles
        selected = valid & (ranked > low) & (ranked <= high if bucket else ranked <= high)
        rows[f"q{bucket + 1:02d}"] = aligned.where(selected).mean(axis=1).mean()
    return pd.Series(rows, name="mean_fwd_ret").to_frame()


def turnover(scores: pd.DataFrame) -> pd.Series:
    """Mean absolute change in cross-sectional percentile rank, per date."""
    ranked = _row_rank(scores)
    return ranked.diff().abs().mean(axis=1)


def autocorrelation(scores: pd.DataFrame, lags: tuple[int, ...] = (1, 5, 21)) -> pd.Series:
    ranked = _row_rank(scores)
    return pd.Series(
        {
            f"autocorr_{lag}": float(_row_corr(ranked, ranked.shift(lag), MIN_NAMES).mean())
            for lag in lags
        }
    )


def coverage(scores: pd.DataFrame) -> pd.Series:
    return scores.notna().sum(axis=1).astype("float64")


@dataclass
class SignalReport:
    """Per-date diagnostics plus the headline numbers that decide go / no-go."""

    name: str
    frame: pd.DataFrame
    summary: dict[str, float | str | None]
    quantiles: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_frame(self) -> pd.DataFrame:
        return self.frame

    def __str__(self) -> str:
        lines = [f"signal: {self.name}"]
        for key, value in self.summary.items():
            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"  {key:<24} {formatted}")
        return "\n".join(lines)


def evaluate(
    signal: Signal,
    panel: FeaturePanel,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    quantiles: int = 10,
) -> SignalReport:
    scores = signal.compute(panel)

    columns: dict[str, pd.Series] = {"coverage": coverage(scores), "turnover": turnover(scores)}
    summary: dict[str, float | str | None] = {"name": signal.name}

    for horizon in horizons:
        forward = panel.label(horizon)
        ic = information_coefficient(scores, forward)
        spread = quantile_spread(scores, forward, quantiles)
        columns[f"ic_{horizon}d"] = ic
        columns[f"spread_{horizon}d"] = spread

        observations = int(ic.notna().sum())
        mean, deviation = float(ic.mean()), float(ic.std())
        summary[f"ic_mean_{horizon}d"] = mean
        summary[f"ic_std_{horizon}d"] = deviation
        # Information ratio of the IC series, annualised the same way a Sharpe is.
        summary[f"ic_ir_{horizon}d"] = (
            mean / deviation * np.sqrt(TRADING_DAYS) if deviation else float("nan")
        )
        summary[f"ic_t_stat_{horizon}d"] = (
            mean / deviation * np.sqrt(observations) if deviation and observations else float("nan")
        )
        summary[f"ic_hit_rate_{horizon}d"] = float((ic > 0).sum() / max(observations, 1))
        summary[f"spread_mean_{horizon}d"] = float(spread.mean())
        summary[f"observations_{horizon}d"] = float(observations)

    summary["turnover_mean"] = float(columns["turnover"].mean())
    summary["coverage_mean"] = float(columns["coverage"].mean())
    summary.update({key: float(value) for key, value in autocorrelation(scores).items()})

    frame = pd.DataFrame(columns)
    frame.index.name = "date"
    buckets = quantile_returns(scores, panel.label(horizons[0]), quantiles)
    return SignalReport(signal.name, frame, summary, buckets)


def report_dir(name: str) -> Path:
    return settings().signals_root / name


def save_report(report: SignalReport, directory: Path | str | None = None) -> Path:
    target = Path(directory) if directory else report_dir(report.name)
    target.mkdir(parents=True, exist_ok=True)

    write_parquet(report.frame.reset_index(), target / "report.parquet")
    write_parquet(report.quantiles.reset_index(names="bucket"), target / "quantiles.parquet")
    write_json(stamp(**report.summary), target / "summary.json")
    _plot(report, target)
    return target


def _plot(report: SignalReport, target: Path) -> None:
    """Best-effort plot bundle; a missing matplotlib must not fail the pipeline."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ic_columns = [name for name in report.frame.columns if name.startswith("ic_")]
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    for column in ic_columns:
        report.frame[column].rolling(63).mean().plot(ax=axes[0], label=column)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title(f"{report.name}: rolling 63-day mean IC")
    axes[0].legend(loc="upper left", fontsize=8)

    report.frame["turnover"].rolling(21).mean().plot(ax=axes[1])
    axes[1].set_title("turnover (21-day mean absolute rank change)")

    if not report.quantiles.empty:
        report.quantiles.iloc[:, 0].plot(kind="bar", ax=axes[2])
        axes[2].set_title("mean forward return by score decile")

    figure.savefig(target / "report.png", dpi=120)
    plt.close(figure)
