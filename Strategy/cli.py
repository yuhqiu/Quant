"""``python -m Strategy backtest --spec Strategy/Library/momentum.toml``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .BackTest import ENGINES
from .oos import read_ledger
from .runner import baselines, build_panel, buy_and_hold, run_spec, sweep
from .spec import StrategySpec
from .walkforward import walk_forward

LIBRARY = Path(__file__).resolve().parent / "Library"


def _load(argument: str) -> StrategySpec:
    path = Path(argument)
    if not path.exists():
        candidate = LIBRARY / f"{argument}.toml"
        if candidate.exists():
            path = candidate
        else:
            raise SystemExit(f"no such strategy spec: {argument}")
    return StrategySpec.load(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="Strategy", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="list bundled strategy specs")

    back = commands.add_parser("backtest", help="run one strategy")
    back.add_argument("--spec", required=True)
    back.add_argument("--engine", choices=list(ENGINES))
    back.add_argument("--start")
    back.add_argument("--end")
    back.add_argument("--name")
    back.add_argument("--no-save", action="store_true")
    back.add_argument("--zero-cost", action="store_true", help="opt in to a frictionless run")

    walk = commands.add_parser("walkforward", help="rolling train/test windows")
    walk.add_argument("--spec", required=True)
    walk.add_argument("--train", type=int, default=756)
    walk.add_argument("--test", type=int, default=252)
    walk.add_argument("--purge", type=int, default=21)
    walk.add_argument("--embargo", type=int, default=5)
    walk.add_argument("--anchored", action="store_true")

    surface = commands.add_parser("sweep", help="run a parameter grid and print the surface")
    surface.add_argument("--spec", required=True)
    surface.add_argument("--param", action="append", required=True, metavar="NAME=V1,V2",
                         help="repeatable, e.g. --param quantiles=3,5,10")
    surface.add_argument("--section", default="constructor", choices=["constructor", "signal", "strategy"])

    base = commands.add_parser("baselines", help="run the null models a strategy must beat")
    base.add_argument("--spec", required=True)
    base.add_argument("--benchmark", default="SPY")

    ledger = commands.add_parser("oos", help="show the out-of-sample touch ledger")
    ledger.add_argument("--spec")

    return parser.parse_args(argv)


def _coerce(text: str) -> object:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    return text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "list":
        for path in sorted(LIBRARY.glob("*.toml")):
            print(f"{path.stem:<24} {path}")
        return 0

    if args.command == "oos":
        ledger = read_ledger()
        if not ledger:
            print("no out-of-sample evaluations recorded")
            return 0
        for entry in ledger.values():
            print(f"{entry['spec_hash'][:12]}  {entry['strategy']:<24} touches={entry['touches']}")
        return 0

    spec = _load(args.spec)

    if args.command == "backtest":
        spec = spec.with_overrides(start=args.start, end=args.end, name=args.name)
        if args.zero_cost:
            from dataclasses import replace

            spec = replace(spec, costs=replace(spec.costs, enabled=False))
            print("WARNING: costs disabled. This result is not tradable.")

        result = run_spec(spec, engine=args.engine, save=not args.no_save)
        _print_metrics(result.metrics)
        if not args.no_save:
            print(f"\nsaved to {result.directory()}")
        return 0

    if args.command == "walkforward":
        panel = build_panel(spec)
        dates = panel.dates
        if spec.start_ts is not None:
            dates = dates[dates >= spec.start_ts]
        if spec.end_ts is not None:
            dates = dates[dates <= spec.end_ts]

        splits = walk_forward(dates, args.train, args.test, purge=args.purge,
                              embargo=args.embargo, anchored=args.anchored)
        rows = []
        for split in splits:
            window = spec.with_overrides(
                start=split.test_start.date().isoformat(),
                end=split.test_end.date().isoformat(),
                name=f"{spec.name}__wf_{split.label}",
            )
            result = run_spec(window, save=False)
            rows.append({**split.as_dict(), **result.metrics})
        frame = pd.DataFrame(rows)
        print(frame.to_string(index=False))
        print(f"\nmean out-of-sample Sharpe: {frame['sharpe'].mean():.3f} over {len(frame)} folds")
        return 0

    if args.command == "sweep":
        grid = {}
        for item in args.param:
            key, _, values = item.partition("=")
            grid[key] = [_coerce(piece) for piece in values.split(",")]
        surface = sweep(spec, grid, section=args.section)
        print(surface.to_string(index=False))
        print(f"\n{len(surface)} configurations evaluated; read the surface, not the maximum")
        return 0

    if args.command == "baselines":
        print(f"benchmark: buy and hold {args.benchmark}")
        _print_metrics(buy_and_hold(args.benchmark, spec).metrics)
        for label, result in baselines(spec).items():
            print(f"\nbaseline: {label}")
            _print_metrics(result.metrics)
        return 0

    return 1


def _print_metrics(metrics: dict) -> None:
    for key, value in metrics.items():
        formatted = f"{value:,.4f}" if isinstance(value, float) else str(value)
        print(f"  {key:<20} {formatted}")


if __name__ == "__main__":
    raise SystemExit(main())
