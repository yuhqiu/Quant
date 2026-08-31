"""Cross-sectional transforms shared by signals and portfolio construction.

All of them operate row-wise: every date is treated independently, because a
cross-sectional strategy compares symbols to each other on the same day, never to
their own past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_BREADTH = 20


def _breadth_mask(frame: pd.DataFrame, minimum: int) -> pd.Series:
    return frame.count(axis=1) >= minimum


def winsorize(
    frame: pd.DataFrame, lower: float = 0.01, upper: float = 0.99
) -> pd.DataFrame:
    """Clip each date to its own quantiles, so one broken print cannot dominate."""
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("expected 0 <= lower < upper <= 1")
    low = frame.quantile(lower, axis=1)
    high = frame.quantile(upper, axis=1)
    return frame.clip(lower=low, upper=high, axis=0)


def demean(frame: pd.DataFrame, minimum: int = MIN_BREADTH) -> pd.DataFrame:
    centered = frame.sub(frame.mean(axis=1), axis=0)
    return centered.where(_breadth_mask(frame, minimum), axis=0)


def zscore(frame: pd.DataFrame, minimum: int = MIN_BREADTH) -> pd.DataFrame:
    centered = frame.sub(frame.mean(axis=1), axis=0)
    scaled = centered.div(frame.std(axis=1).replace(0.0, np.nan), axis=0)
    return scaled.where(_breadth_mask(frame, minimum), axis=0)


def rank_normalize(frame: pd.DataFrame, minimum: int = MIN_BREADTH) -> pd.DataFrame:
    """Percentile rank mapped to [-1, 1]; immune to outliers by construction."""
    ranked = frame.rank(axis=1, pct=True)
    return (2.0 * ranked - 1.0).where(_breadth_mask(frame, minimum), axis=0)


def group_neutral(frame: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Subtract each group's own daily mean, e.g. sector neutralisation."""
    labels = groups.reindex(frame.columns)
    known = labels.dropna()
    if known.empty:
        return demean(frame)

    result = frame.copy()
    for name in known.unique():
        members = [column for column in known.index[known == name] if column in frame.columns]
        block = frame[members]
        result[members] = block.sub(block.mean(axis=1), axis=0)

    unknown = [column for column in frame.columns if column not in known.index]
    if unknown:
        result[unknown] = np.nan
    return result


def beta_neutral(frame: pd.DataFrame, beta: pd.DataFrame) -> pd.DataFrame:
    """Remove the component of the score explained by beta, date by date."""
    exposures = beta.reindex(index=frame.index, columns=frame.columns)
    valid = frame.notna() & exposures.notna()
    scores = frame.where(valid)
    betas = exposures.where(valid)

    centered_scores = scores.sub(scores.mean(axis=1), axis=0)
    centered_betas = betas.sub(betas.mean(axis=1), axis=0)

    covariance = (centered_scores * centered_betas).sum(axis=1)
    variance = (centered_betas**2).sum(axis=1).replace(0.0, np.nan)
    slope = covariance / variance
    return centered_scores.sub(centered_betas.mul(slope, axis=0))


def neutralize(
    frame: pd.DataFrame,
    method: str = "demean",
    groups: pd.Series | None = None,
    beta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if method == "none":
        return frame
    if method == "demean":
        return demean(frame)
    if method == "zscore":
        return zscore(frame)
    if method == "rank":
        return rank_normalize(frame)
    if method == "group":
        if groups is None:
            raise ValueError("group neutralisation needs a group mapping")
        return group_neutral(frame, groups)
    if method == "beta":
        if beta is None:
            raise ValueError("beta neutralisation needs a beta matrix")
        return beta_neutral(frame, beta)
    raise ValueError(f"unknown neutralisation method {method!r}")
