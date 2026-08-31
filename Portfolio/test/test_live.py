"""Portfolio construction against the real feature panel."""

from __future__ import annotations

import os

import pytest

from Common.types import Partition
from MetricsGeneration import read_manifest
from Portfolio import build_targets
from Portfolio.base import MarketContext
from Portfolio.constraints import build_chain
from Portfolio.constructors import QuantileLongShort, TopNEqualWeight
from Signals import FeaturePanel, get_signal

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("QUANT_LIVE_TESTS") != "1",
        reason="set QUANT_LIVE_TESTS=1 to run tests against the real panel",
    ),
]

PARTITION = Partition("US", "stock", "1d")


@pytest.fixture(scope="module")
def scored():
    manifest = read_manifest(PARTITION.metrics_dir)
    if manifest is None:
        pytest.skip("no feature panel built")

    panel = FeaturePanel(
        partition=PARTITION, symbols=tuple(manifest["symbol_list"][:400]), start="2018-01-01"
    )
    signal = get_signal("momentum_12_1")
    scores = signal.compute(panel)
    context = MarketContext.from_panel(panel)
    context.tradable = signal.tradability.mask(panel)
    return scores, context


class TestRealBook:
    def test_long_short_book_is_dollar_neutral(self, scored):
        scores, context = scored
        targets = build_targets(scores, context, QuantileLongShort(5), rebalance="weekly")
        live = targets.dropna(how="all")
        assert live.sum(axis=1).abs().max() < 1e-9

    def test_gross_exposure_is_respected(self, scored):
        scores, context = scored
        targets = build_targets(scores, context, QuantileLongShort(5, gross=1.0), rebalance="weekly")
        live = targets.dropna(how="all")
        assert live.abs().sum(axis=1).max() <= 1.0 + 1e-9

    def test_untradable_names_are_never_held(self, scored):
        scores, context = scored
        targets = build_targets(
            scores, context, TopNEqualWeight(50), constraints=build_chain({"tradable_only": {}}),
            rebalance="monthly",
        )
        held = targets.dropna(how="all") != 0.0
        allowed = context.tradable.reindex(index=held.index, columns=held.columns).fillna(False)
        assert not (held & ~allowed).to_numpy().any()

    def test_constraint_chain_caps_concentration(self, scored):
        scores, context = scored
        targets = build_targets(
            scores, context, QuantileLongShort(5),
            constraints=build_chain({"max_weight_per_name": 0.02}),
            rebalance="weekly",
        )
        assert targets.abs().max().max() <= 0.02 + 1e-9

    def test_turnover_cap_binds(self, scored):
        scores, context = scored
        capped = build_targets(
            scores, context, QuantileLongShort(5), rebalance="weekly", max_turnover=0.1
        ).dropna(how="all")
        moves = capped.diff().abs().sum(axis=1).iloc[1:]
        assert moves.max() <= 0.1 + 1e-9
