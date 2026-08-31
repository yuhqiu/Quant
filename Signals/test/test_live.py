"""Signals against the real feature panel. Skipped unless ``QUANT_LIVE_TESTS=1``."""

from __future__ import annotations

import os

import pytest

from Common.types import Partition
from MetricsGeneration import read_manifest
from Signals import FeaturePanel, evaluate, get_signal, signal_names

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("QUANT_LIVE_TESTS") != "1",
        reason="set QUANT_LIVE_TESTS=1 to run tests against the real panel",
    ),
]

PARTITION = Partition("US", "stock", "1d")


@pytest.fixture(scope="module")
def panel() -> FeaturePanel:
    manifest = read_manifest(PARTITION.metrics_dir)
    if manifest is None:
        pytest.skip("no feature panel built")
    symbols = tuple(manifest["symbol_list"][:400])
    return FeaturePanel(partition=PARTITION, symbols=symbols, start="2015-01-01")


class TestPanelAccess:
    def test_metrics_load_lazily(self, panel):
        assert not panel._cache
        panel.get("close")
        assert set(panel._cache) == {"close"}

    def test_labels_resolve_from_the_label_directory(self, panel):
        assert panel.label(21).shape == panel.get("close").shape

    def test_missing_metric_raises(self, panel):
        with pytest.raises(KeyError):
            panel.get("not_a_real_metric")


class TestRealSignals:
    @pytest.mark.parametrize("name", ["momentum_12_1", "reversal_5d", "low_vol_20d"])
    def test_signal_produces_scores(self, panel, name):
        scores = get_signal(name).compute(panel)
        assert scores.notna().to_numpy().sum() > 0
        assert scores.index.equals(panel.dates)

    def test_momentum_beats_noise_on_information_coefficient(self, panel):
        momentum = evaluate(get_signal("momentum_12_1"), panel, (21,))
        noise = evaluate(get_signal("random", seed=0), panel, (21,))
        assert abs(momentum.summary["ic_mean_21d"]) > abs(noise.summary["ic_mean_21d"])

    def test_momentum_turnover_is_lower_than_reversal(self, panel):
        momentum = evaluate(get_signal("momentum_12_1"), panel, (5,))
        reversal = evaluate(get_signal("reversal_5d"), panel, (5,))
        assert momentum.summary["turnover_mean"] < reversal.summary["turnover_mean"]

    def test_every_registered_signal_runs(self, panel):
        for name in signal_names():
            params = {"metric": "ret_5d"} if name == "metric" else {}
            scores = get_signal(name, **params).compute(panel)
            assert scores.shape[0] == len(panel.dates)

    def test_tradability_filter_shrinks_coverage(self, panel):
        wide = get_signal("momentum_12_1", tradability=None).compute(panel)
        narrow = get_signal(
            "momentum_12_1", tradability={"min_price": 10.0, "min_dollar_volume": 5e7}
        ).compute(panel)
        assert narrow.notna().to_numpy().sum() < wide.notna().to_numpy().sum()
