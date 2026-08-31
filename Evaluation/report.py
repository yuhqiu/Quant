"""Turn a saved backtest into a judgement: an HTML tearsheet and machine-readable metrics.

Results are read straight from the run directory, so this module has no
dependency on the engine that wrote them.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from Common.config import settings
from Common.io import read_json, read_parquet, write_json
from Common.provenance import stamp

from .metrics import drawdown, monthly_returns, performance, sharpe

EQUITY_FILE = "equity.parquet"
POSITIONS_FILE = "positions.parquet"
TRADES_FILE = "trades.parquet"
SPEC_FILE = "spec.json"
REPORT_FILE = "report.html"
METRICS_FILE = "evaluation.json"


@dataclass
class RunArtifacts:
    """The four files every run writes, loaded back into memory."""

    path: Path
    equity: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    spec: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.spec.get("name", self.path.parent.name))

    @property
    def run_id(self) -> str:
        return self.path.name

    @property
    def returns(self) -> pd.Series:
        return self.equity["equity"].pct_change().dropna()


def load_run(path: Path | str) -> RunArtifacts:
    directory = Path(path)
    equity = read_parquet(directory / EQUITY_FILE)
    equity["date"] = pd.to_datetime(equity["date"], utc=True)
    equity = equity.set_index("date").sort_index()

    return RunArtifacts(
        path=directory,
        equity=equity,
        trades=read_parquet(directory / TRADES_FILE),
        positions=read_parquet(directory / POSITIONS_FILE),
        spec=read_json(directory / SPEC_FILE) if (directory / SPEC_FILE).exists() else {},
    )


def find_run(strategy: str, run_id: str = "latest", root: Path | str | None = None) -> Path:
    base = Path(root) if root else settings().backtests_root
    directory = base / strategy
    if not directory.is_dir():
        raise FileNotFoundError(f"no backtests for {strategy!r} under {base}")
    if run_id != "latest":
        return directory / run_id
    runs = sorted(path for path in directory.iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"no runs recorded for {strategy!r}")
    return runs[-1]


def evaluate(
    run: RunArtifacts,
    benchmark: pd.Series | None = None,
    trials: int = 1,
) -> dict[str, Any]:
    return performance(
        equity=run.equity,
        trades=run.trades,
        benchmark=benchmark,
        positions=run.positions,
        trials=trials,
    )


def report(
    run: RunArtifacts,
    benchmark: pd.Series | None = None,
    trials: int = 1,
    output: Path | str | None = None,
) -> Path:
    """Write ``report.html`` and ``evaluation.json`` next to the run."""
    summary = evaluate(run, benchmark, trials)
    target = Path(output) if output else run.path
    target.mkdir(parents=True, exist_ok=True)

    write_json(stamp(**summary), target / METRICS_FILE)
    html = _render(run, summary, benchmark)
    (target / REPORT_FILE).write_text(html, encoding="utf-8")
    return target / REPORT_FILE


def compare(paths: list[Path | str]) -> pd.DataFrame:
    """Side-by-side headline metrics across runs."""
    rows = []
    for path in paths:
        run = load_run(path)
        rows.append({"strategy": run.name, "run_id": run.run_id, **evaluate(run)})
    frame = pd.DataFrame(rows)
    return frame.set_index(["strategy", "run_id"]) if not frame.empty else frame


# --- rendering ------------------------------------------------------------
_STYLE = """
body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 2rem auto; max-width: 1100px;
       color: #1c1c1c; background: #fbfbfb; }
h1 { margin-bottom: 0; } h2 { margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }
.sub { color: #666; margin-top: .2rem; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { padding: .35rem .6rem; text-align: right; border-bottom: 1px solid #eee; }
th:first-child, td:first-child { text-align: left; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: .6rem; }
.card { background: #fff; border: 1px solid #e6e6e6; border-radius: 6px; padding: .7rem 1rem; }
.card .label { color: #777; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.card .value { font-size: 22px; font-weight: 600; }
.neg { color: #b3261e; } .pos { color: #146c2e; }
img { width: 100%; border: 1px solid #e6e6e6; border-radius: 6px; background: #fff; }
"""

_HEADLINE = (
    ("total_return", "Total return", "pct"),
    ("cagr", "CAGR", "pct"),
    ("volatility", "Volatility", "pct"),
    ("sharpe", "Sharpe", "num"),
    ("sortino", "Sortino", "num"),
    ("calmar", "Calmar", "num"),
    ("max_drawdown", "Max drawdown", "pct"),
    ("deflated_sharpe", "Deflated Sharpe", "num"),
    ("annual_turnover", "Annual turnover", "num"),
    ("cost_ratio", "Cost / gross P&L", "pct"),
)


def _format(value: Any, kind: str) -> str:
    if not isinstance(value, (int, float)) or value != value:
        return "n/a"
    if kind == "pct":
        return f"{value * 100:,.2f}%"
    return f"{value:,.3f}"


def _render(run: RunArtifacts, summary: dict[str, Any], benchmark: pd.Series | None) -> str:
    cards = "".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value {"neg" if isinstance(summary.get(key), float) and summary.get(key, 0) < 0 else "pos"}">'
        f"{_format(summary.get(key), kind)}</div></div>"
        for key, label, kind in _HEADLINE
    )

    figures = "".join(
        f"<h2>{title}</h2><img src='data:image/png;base64,{image}'/>"
        for title, image in _figures(run, benchmark)
    )

    rows = "".join(
        f"<tr><td>{key}</td><td>{_format(value, 'num') if isinstance(value, float) else value}</td></tr>"
        for key, value in sorted(summary.items())
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{run.name} - {run.run_id}</title>
<style>{_STYLE}</style></head><body>
<h1>{run.name}</h1>
<p class="sub">run {run.run_id} &middot; {summary.get('start')} to {summary.get('end')}
 &middot; engine {run.spec.get('engine', 'vectorised')}
 &middot; commit {str(run.spec.get('git_commit') or 'unknown')[:12]}</p>
<div class="grid">{cards}</div>
{figures}
<h2>All metrics</h2>
<table><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""


def _figures(run: RunArtifacts, benchmark: pd.Series | None) -> list[tuple[str, str]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    equity = run.equity["equity"]
    returns = run.returns
    charts: list[tuple[str, str]] = []

    figure, axis = plt.subplots(figsize=(11, 4))
    (equity / equity.iloc[0]).plot(ax=axis, logy=True, label=run.name)
    if benchmark is not None and len(benchmark):
        (1.0 + benchmark.reindex(equity.index).fillna(0.0)).cumprod().plot(
            ax=axis, label="benchmark", alpha=0.7
        )
        axis.legend()
    axis.set_ylabel("growth of 1 (log)")
    charts.append(("Equity curve", _encode(figure, plt)))

    figure, axis = plt.subplots(figsize=(11, 2.6))
    under = drawdown(equity)
    axis.fill_between(under.index, under.to_numpy(), 0.0, color="#b3261e", alpha=0.4)
    axis.set_ylabel("drawdown")
    charts.append(("Underwater", _encode(figure, plt)))

    figure, axis = plt.subplots(figsize=(11, 2.6))
    rolling = returns.rolling(252).apply(lambda window: sharpe(pd.Series(window)), raw=False)
    rolling.plot(ax=axis)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("rolling 252d Sharpe")
    charts.append(("Rolling Sharpe", _encode(figure, plt)))

    monthly = monthly_returns(returns)
    if len(monthly) > 1:
        table = monthly.to_frame("ret")
        table["year"] = table.index.year
        table["month"] = table.index.month
        grid = table.pivot_table(index="year", columns="month", values="ret")
        figure, axis = plt.subplots(figsize=(11, max(2.4, 0.32 * len(grid))))
        image = axis.imshow(grid.to_numpy(), cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.15)
        axis.set_xticks(range(len(grid.columns)), [str(c) for c in grid.columns])
        axis.set_yticks(range(len(grid.index)), [str(i) for i in grid.index])
        figure.colorbar(image, ax=axis, shrink=0.8)
        charts.append(("Monthly returns", _encode(figure, plt)))

    figure, axis = plt.subplots(figsize=(11, 2.6))
    run.equity[["long_value", "short_value"]].div(equity, axis=0).plot(ax=axis)
    axis.set_ylabel("exposure / equity")
    charts.append(("Exposure", _encode(figure, plt)))

    if not run.trades.empty:
        figure, axis = plt.subplots(figsize=(11, 2.6))
        costs = run.trades.groupby("date")[["commission", "spread", "slippage"]].sum().cumsum()
        costs.plot(ax=axis)
        axis.set_ylabel("cumulative cost")
        charts.append(("Cost breakdown", _encode(figure, plt)))

    return charts


def _encode(figure, plt) -> str:
    buffer = io.BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format="png", dpi=110)
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
