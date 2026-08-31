"""``python -m Evaluation report --strategy momentum --run latest``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from Common.config import settings

from .report import compare, evaluate, find_run, load_run, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="Evaluation", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="list recorded backtest runs")

    build = commands.add_parser("report", help="write the HTML tearsheet for one run")
    build.add_argument("--strategy", required=True)
    build.add_argument("--run", default="latest")
    build.add_argument("--benchmark", help="path to a run directory used as the benchmark")
    build.add_argument("--trials", type=int, default=1, help="configurations evaluated, for the deflated Sharpe")
    build.add_argument("--output", type=Path)

    table = commands.add_parser("compare", help="side-by-side metrics across runs")
    table.add_argument("runs", nargs="+", help="strategy names or run directories")

    return parser.parse_args(argv)


def _resolve(reference: str) -> Path:
    path = Path(reference)
    if path.is_dir():
        return path
    strategy, _, run_id = reference.partition(":")
    return find_run(strategy, run_id or "latest")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = settings().backtests_root

    if args.command == "list":
        if not root.is_dir():
            print(f"no backtests under {root}")
            return 1
        for strategy in sorted(path for path in root.iterdir() if path.is_dir()):
            runs = sorted(path.name for path in strategy.iterdir() if path.is_dir())
            print(f"{strategy.name:<28} {len(runs):>3} run(s)  latest={runs[-1] if runs else '-'}")
        return 0

    if args.command == "report":
        run = load_run(find_run(args.strategy, args.run))
        benchmark = None
        if args.benchmark:
            benchmark = load_run(_resolve(args.benchmark)).returns

        summary = evaluate(run, benchmark, args.trials)
        for key, value in summary.items():
            formatted = f"{value:,.4f}" if isinstance(value, float) else str(value)
            print(f"  {key:<22} {formatted}")

        path = report(run, benchmark, args.trials, args.output)
        print(f"\nwritten to {path}")
        return 0

    if args.command == "compare":
        frame = compare([_resolve(item) for item in args.runs])
        if frame.empty:
            print("nothing to compare")
            return 1
        columns = [
            name
            for name in ("total_return", "cagr", "volatility", "sharpe", "max_drawdown",
                         "annual_turnover", "cost_ratio", "deflated_sharpe")
            if name in frame.columns
        ]
        with pd.option_context("display.float_format", lambda value: f"{value:,.4f}"):
            print(frame[columns].to_string())
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
