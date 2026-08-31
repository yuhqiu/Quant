"""``python -m Signals evaluate --signal momentum_12_1``."""

from __future__ import annotations

import argparse

from Common.types import ASSET_CLASSES, Partition

from .library import get_signal, signal_names
from .panel import FeaturePanel
from .report import evaluate, save_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="Signals", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="list registered signals")

    check = commands.add_parser("evaluate", help="score a signal and write its quality report")
    check.add_argument("--signal", required=True, choices=signal_names())
    check.add_argument("--metric", help="metric name when --signal metric is used")
    check.add_argument("--sign", type=float, default=1.0)
    check.add_argument("--region", default="US")
    check.add_argument("--asset-class", default="stock", choices=list(ASSET_CLASSES))
    check.add_argument("--interval", default="1d")
    check.add_argument("--symbols", nargs="+")
    check.add_argument("--start")
    check.add_argument("--end")
    check.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 21])
    check.add_argument("--quantiles", type=int, default=10)
    check.add_argument("--neutralization", default="zscore")
    check.add_argument("--min-price", type=float, default=5.0)
    check.add_argument("--min-dollar-volume", type=float, default=1_000_000.0)
    check.add_argument("--no-save", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "list":
        for name in signal_names():
            print(name)
        return 0

    params: dict[str, object] = {
        "neutralization": args.neutralization,
        "tradability": {
            "min_price": args.min_price,
            "min_dollar_volume": args.min_dollar_volume,
        },
    }
    if args.signal == "metric":
        if not args.metric:
            print("--metric is required when --signal metric is used")
            return 2
        params["metric"] = args.metric
        params["sign"] = args.sign

    signal = get_signal(args.signal, **params)
    panel = FeaturePanel(
        partition=Partition(args.region, args.asset_class, args.interval),
        symbols=tuple(args.symbols) if args.symbols else None,
        start=args.start,
        end=args.end,
    )

    report = evaluate(signal, panel, tuple(args.horizons), args.quantiles)
    print(report)
    if not args.no_save:
        print(f"\nwritten to {save_report(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
