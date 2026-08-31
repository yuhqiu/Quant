"""End-to-end strategy runs against the real panel."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from Common.types import Partition
from MetricsGeneration import read_manifest
from Strategy import StrategySpec, run_spec
from Strategy.BackTest.costs import ZERO_COST

pytestmark = [
    pytest.mark.live,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("QUANT_LIVE_TESTS") != "1",
        reason="set QUANT_LIVE_TESTS=1 to run end-to-end strategy tests",
    ),
]

PARTITION = Partition("US", "stock", "1d")


@pytest.fixture(scope="module")
def spec(request) -> StrategySpec:
    if read_manifest(PARTITION.metrics_dir) is None:
        pytest.skip("no feature panel built")
    root = request.config.rootpath
    loaded = StrategySpec.load(root / "Strategy" / "Library" / "momentum.toml")
    return loaded.with_overrides(start="2018-01-01", end="2022-12-31")


@pytest.fixture(scope="module")
def result(spec):
    return run_spec(spec, save=False)


class TestEndToEnd:
    def test_the_run_produces_a_full_result(self, result):
        assert len(result.equity) > 500
        assert not result.trades.empty
        assert not result.positions.empty

    def test_accounting_invariants_hold(self, result):
        assert result.check_invariants() == []

    def test_costs_were_charged(self, result):
        assert result.cost_summary()["total"] > 0.0

    def test_exposure_matches_the_specification(self, result, spec):
        leverage = result.equity["leverage"].dropna()
        assert leverage.max() < 1.5
        assert abs(result.equity["net"].div(result.equity["equity"]).mean()) < 0.1

    def test_costs_reduce_the_return(self, spec):
        free = run_spec(replace(spec, costs=ZERO_COST), save=False)
        charged = run_spec(spec, save=False)
        assert charged.final_equity < free.final_equity

    def test_reruns_are_byte_identical(self, spec):
        import pandas as pd

        first = run_spec(spec, save=False, run_id="fixed")
        second = run_spec(spec, save=False, run_id="fixed")
        pd.testing.assert_frame_equal(first.equity, second.equity)

    def test_random_signal_does_not_beat_momentum_by_much(self, spec):
        null = replace(
            spec,
            name="null",
            signal=replace(spec.signal, name="random", params={"seed": 3}),
        )
        assert run_spec(null, save=False).metrics["sharpe"] < 1.0


class TestArtifacts:
    def test_saving_writes_every_file(self, spec, tmp_path):
        result = run_spec(spec, save=True, root=tmp_path)
        target = result.directory(tmp_path)
        for name in ("equity.parquet", "positions.parquet", "trades.parquet",
                     "metrics.json", "spec.json"):
            assert (target / name).exists(), name

    def test_the_spec_hash_is_recorded(self, spec, tmp_path):
        result = run_spec(spec, save=False)
        assert result.spec["spec_hash"] == spec.hash
