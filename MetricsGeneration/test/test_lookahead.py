"""Lookahead detection.

Recompute every feature on a truncated history: the value at date ``T`` must not
move when the future is removed. Any feature that changes is leaking, and this
runs across the whole registry rather than a hand-picked sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from MetricsGeneration.indicators import (
    LABEL_COLUMNS,
    compute_labels,
    compute_metrics,
    metric_names,
    min_periods,
)

CUTS = (120, 250, 380)


def _mismatch(full: pd.Series, truncated: pd.Series, warmup: int) -> float:
    """Largest relative difference over the region both series claim to define."""
    start = min(warmup, len(truncated) - 1)
    a = full.iloc[start : len(truncated)].to_numpy(dtype=float)
    b = truncated.iloc[start:].to_numpy(dtype=float)

    both_nan = np.isnan(a) & np.isnan(b)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")

    a, b = a[~both_nan], b[~both_nan]
    if a.size == 0:
        return 0.0
    scale = np.maximum(np.abs(a), 1.0)
    return float(np.max(np.abs(a - b) / scale))


class TestLookahead:
    @pytest.mark.parametrize("cut", CUTS)
    def test_no_feature_moves_when_the_future_is_removed(self, bars, cut):
        full = compute_metrics(bars)
        truncated = compute_metrics(bars.iloc[:cut])

        leaking = {
            column: _mismatch(full[column], truncated[column], min_periods(column))
            for column in full.columns
        }
        offenders = {name: error for name, error in leaking.items() if error > 1e-9}
        assert not offenders, f"features changed when future data was removed: {offenders}"

    def test_every_registered_metric_is_covered(self, bars):
        produced = set(compute_metrics(bars).columns)
        assert produced == set(metric_names())

    def test_the_detector_actually_detects(self, bars):
        """Labels are forward-looking by construction, so they must fail the test."""
        cut = 250
        full = compute_labels(bars)
        truncated = compute_labels(bars.iloc[:cut])

        errors = [
            _mismatch(full[column], truncated[column], 0) for column in LABEL_COLUMNS
        ]
        assert max(errors) > 1e-9, "a leaking column passed the lookahead test"

    def test_labels_live_in_their_own_namespace(self):
        assert not set(LABEL_COLUMNS) & set(metric_names())
