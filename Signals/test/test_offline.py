"""Signals: transforms, tradability, and IC diagnostics on synthetic panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Signals.base import TradabilityFilter
from Signals.combine import RankCombine, ZScoreCombine
from Signals.library import MetricSignal, Momentum12_1, RandomSignal, get_signal, signal_names
from Signals.panel import FeaturePanel
from Signals.report import (
    autocorrelation,
    coverage,
    evaluate,
    information_coefficient,
    quantile_spread,
    turnover,
)
from Signals.transforms import (
    beta_neutral,
    demean,
    group_neutral,
    rank_normalize,
    winsorize,
    zscore,
)


@pytest.fixture
def panel(tmp_path, panel_frames) -> FeaturePanel:
    from Common.io import matrix_path, write_matrix

    metrics_dir, labels_dir = tmp_path / "metrics", tmp_path / "labels"
    for name, frame in panel_frames.items():
        target = labels_dir if name.startswith("fwd_ret_") else metrics_dir
        write_matrix(frame, matrix_path(target, name))
    return FeaturePanel(metrics_dir=metrics_dir, labels_dir=labels_dir)


@pytest.fixture
def wide() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=4, tz="UTC")
    return pd.DataFrame(
        np.array([[1.0, 2, 3, 100], [4.0, 5, 6, 7], [np.nan, 1, 2, 3], [0.0, 0, 0, 0]]),
        index=index,
        columns=list("ABCD"),
    )


class TestTransforms:
    def test_demean_sums_to_zero(self, wide):
        result = demean(wide, minimum=1)
        assert result.sum(axis=1).abs().max() == pytest.approx(0.0)

    def test_zscore_has_unit_dispersion(self, wide):
        result = zscore(wide, minimum=1)
        assert result.iloc[0].std(ddof=1) == pytest.approx(1.0)

    def test_zscore_blanks_dates_below_the_breadth_floor(self, wide):
        assert zscore(wide, minimum=10).isna().all().all()

    def test_rank_normalize_spans_minus_one_to_one(self, wide):
        result = rank_normalize(wide, minimum=1)
        assert result.iloc[0].min() == pytest.approx(-0.5)
        assert result.iloc[0].max() == pytest.approx(1.0)

    def test_winsorize_pulls_in_the_outlier(self, wide):
        clipped = winsorize(wide, 0.0, 0.75)
        assert clipped.iloc[0].max() < 100.0

    def test_constant_row_zscores_to_nan(self, wide):
        assert zscore(wide, minimum=1).iloc[3].isna().all()

    def test_group_neutral_zeroes_each_group(self, wide):
        groups = pd.Series({"A": "x", "B": "x", "C": "y", "D": "y"})
        result = group_neutral(wide, groups)
        assert result.loc[:, ["A", "B"]].sum(axis=1).abs().max() == pytest.approx(0.0)
        assert result.loc[:, ["C", "D"]].sum(axis=1).abs().max() == pytest.approx(0.0)

    def test_beta_neutral_removes_the_beta_tilt(self):
        index = pd.date_range("2024-01-01", periods=3, tz="UTC")
        scores = pd.DataFrame(np.array([[1.0, 2, 3, 4]] * 3), index=index, columns=list("ABCD"))
        betas = pd.DataFrame(np.array([[0.5, 1.0, 1.5, 2.0]] * 3), index=index, columns=list("ABCD"))
        result = beta_neutral(scores, betas)
        # Scores are an exact linear function of beta, so nothing should survive.
        assert result.abs().to_numpy().max() == pytest.approx(0.0, abs=1e-12)


class TestTradability:
    def test_penny_stocks_are_excluded(self, panel):
        allowed = TradabilityFilter(min_price=1e9, min_dollar_volume=0.0, require_history=0).mask(panel)
        assert not allowed.to_numpy().any()

    def test_a_permissive_filter_lets_everything_through(self, panel):
        allowed = TradabilityFilter(
            min_price=0.0, min_dollar_volume=0.0, max_stale_fraction=1.0, require_history=0
        ).mask(panel)
        assert allowed.to_numpy().sum() > 0

    def test_history_requirement_blanks_the_warm_up(self, panel):
        allowed = TradabilityFilter(
            min_price=0.0, min_dollar_volume=0.0, require_history=100
        ).mask(panel)
        assert not allowed.iloc[:99].to_numpy().any()


class TestSignals:
    def test_registry_is_complete(self):
        assert {"momentum_12_1", "reversal_5d", "low_vol_20d", "random"} <= set(signal_names())

    def test_unknown_signal_is_rejected(self):
        with pytest.raises(ValueError, match="unknown signal"):
            get_signal("does_not_exist")

    def test_signal_writes_no_files(self, panel, tmp_path):
        before = set(tmp_path.rglob("*"))
        Momentum12_1().compute(panel)
        assert set(tmp_path.rglob("*")) == before

    def test_signal_output_is_aligned_to_the_panel(self, panel):
        scores = Momentum12_1().compute(panel)
        assert list(scores.columns) == panel.universe
        assert scores.index.equals(panel.dates)

    def test_random_signal_is_reproducible(self, panel):
        first = RandomSignal(seed=42).compute(panel)
        second = RandomSignal(seed=42).compute(panel)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seeds_differ(self, panel):
        first = RandomSignal(seed=1).compute(panel)
        second = RandomSignal(seed=2).compute(panel)
        assert not first.equals(second)

    def test_panel_reads_only_what_is_asked_for(self, panel):
        Momentum12_1(tradability=None).compute(panel)
        assert set(panel._cache) == {"mom_12_1"}

    def test_missing_metric_is_reported_clearly(self, panel):
        with pytest.raises(KeyError, match="missing metrics"):
            MetricSignal(metric="not_a_metric", tradability=None).compute(panel)

    def test_composite_blends_components(self, panel):
        composite = RankCombine(
            [Momentum12_1(tradability=None), MetricSignal(metric="ret_5d", sign=-1.0, tradability=None)],
            weights=[0.5, 0.5],
        )
        blended = composite.compute(panel)
        assert blended.shape == panel.get("close").shape

    def test_composite_weights_are_normalised(self, panel):
        composite = ZScoreCombine([Momentum12_1(tradability=None)], weights=[7.0])
        assert composite.weights == [1.0]


class TestReport:
    def test_a_perfect_signal_has_ic_one(self):
        index = pd.date_range("2024-01-01", periods=30, tz="UTC")
        columns = [f"S{i:02d}" for i in range(25)]
        generator = np.random.default_rng(0)
        forward = pd.DataFrame(generator.normal(size=(30, 25)), index=index, columns=columns)
        ic = information_coefficient(forward, forward)
        assert ic.mean() == pytest.approx(1.0)

    def test_an_inverted_signal_has_ic_minus_one(self):
        index = pd.date_range("2024-01-01", periods=30, tz="UTC")
        columns = [f"S{i:02d}" for i in range(25)]
        forward = pd.DataFrame(
            np.random.default_rng(1).normal(size=(30, 25)), index=index, columns=columns
        )
        assert information_coefficient(-forward, forward).mean() == pytest.approx(-1.0)

    def test_ic_is_blank_when_breadth_is_too_thin(self):
        index = pd.date_range("2024-01-01", periods=5, tz="UTC")
        frame = pd.DataFrame(np.random.default_rng(2).normal(size=(5, 3)), index=index)
        assert information_coefficient(frame, frame).isna().all()

    def test_perfect_signal_has_positive_quantile_spread(self):
        index = pd.date_range("2024-01-01", periods=30, tz="UTC")
        columns = [f"S{i:02d}" for i in range(40)]
        forward = pd.DataFrame(
            np.random.default_rng(3).normal(size=(30, 40)), index=index, columns=columns
        )
        assert quantile_spread(forward, forward, 10).mean() > 0.0

    def test_a_constant_signal_has_zero_turnover(self):
        index = pd.date_range("2024-01-01", periods=10, tz="UTC")
        frame = pd.DataFrame(
            np.tile(np.arange(25.0), (10, 1)), index=index, columns=[f"S{i:02d}" for i in range(25)]
        )
        assert turnover(frame).iloc[1:].abs().max() == pytest.approx(0.0)
        assert autocorrelation(frame)["autocorr_1"] == pytest.approx(1.0)

    def test_coverage_counts_live_names(self, panel):
        scores = Momentum12_1(tradability=None).compute(panel)
        assert coverage(scores).max() == len(panel.universe)

    def test_evaluate_produces_every_horizon(self, panel):
        report = evaluate(Momentum12_1(tradability=None), panel, (1, 5, 21))
        for horizon in (1, 5, 21):
            assert f"ic_mean_{horizon}d" in report.summary
            assert f"ic_{horizon}d" in report.frame.columns
        assert "turnover_mean" in report.summary

    def test_random_signal_has_no_edge(self, panel):
        report = evaluate(RandomSignal(seed=5, tradability=None), panel, (5,))
        assert abs(report.summary["ic_mean_5d"]) < 0.05

    def test_report_can_be_saved(self, panel, tmp_path):
        from Signals.report import save_report

        report = evaluate(RandomSignal(seed=1, tradability=None), panel, (1,))
        target = save_report(report, tmp_path / "out")
        assert (target / "report.parquet").exists()
        assert (target / "summary.json").exists()
