"""Evaluation against real saved runs."""

from __future__ import annotations

import os

import pytest

from Common.config import settings
from Evaluation import compare, evaluate, find_run, load_run, report

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("QUANT_LIVE_TESTS") != "1",
        reason="set QUANT_LIVE_TESTS=1 to evaluate real runs",
    ),
]


@pytest.fixture(scope="module")
def run():
    root = settings().backtests_root
    if not root.is_dir():
        pytest.skip("no backtests recorded")
    strategies = [path for path in root.iterdir() if path.is_dir() and any(path.iterdir())]
    if not strategies:
        pytest.skip("no backtests recorded")
    return load_run(find_run(strategies[0].name))


class TestRealRun:
    def test_artifacts_load(self, run):
        assert len(run.equity) > 0
        assert "equity" in run.equity.columns

    def test_metrics_are_finite(self, run):
        summary = evaluate(run)
        for key in ("total_return", "cagr", "volatility", "sharpe", "max_drawdown"):
            assert summary[key] == summary[key], key

    def test_drawdown_is_not_positive(self, run):
        assert evaluate(run)["max_drawdown"] <= 0.0

    def test_report_renders(self, run, tmp_path):
        path = report(run, output=tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "<html" in text
        assert "Equity curve" in text

    def test_compare_returns_one_row_per_run(self, run):
        table = compare([run.path])
        assert len(table) == 1
