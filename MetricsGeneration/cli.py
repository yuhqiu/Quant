"""Command line for the metrics stage: ``python -m MetricsGeneration build``."""

from __future__ import annotations

import argparse
from pathlib import Path

from Common.io import available_metrics, matrix_path, read_matrix
from Common.types import ASSET_CLASSES, Partition

from .metrics_generation import build, read_manifest


def _partition(args: argparse.Namespace) -> Partition:
    return Partition(
        region=args.region, asset_class=args.asset_class, interval=args.interval
    )


def _add_partition_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--region", default="US")
    parser.add_argument("--asset-class", default="stock", choices=list(ASSET_CLASSES))
    parser.add_argument("--interval", default="1d")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="MetricsGeneration", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    builder = commands.add_parser("build", help="rebuild the feature panel")
    _add_partition_flags(builder)
    builder.add_argument("--metrics-out", type=Path)
    builder.add_argument("--labels-out", type=Path)
    builder.add_argument("--symbols", nargs="+", help="restrict the build to these tickers")
    builder.add_argument("--symbols-file", type=Path, help="one ticker per line")
    builder.add_argument("--limit", type=int, help="use only the first N symbols")
    builder.add_argument("--batch-size", type=int, default=100)
    builder.add_argument("--workers", type=int, default=8)
    builder.add_argument("--no-labels", action="store_true")
    builder.add_argument("--no-cross-section", action="store_true")
    builder.add_argument("--incremental", action="store_true", help="skip when the lake has no new bars")
    builder.add_argument("--keep-staging", action="store_true")

    status = commands.add_parser("status", help="show the current panel manifest")
    _add_partition_flags(status)

    listing = commands.add_parser("list", help="list built metrics")
    _add_partition_flags(listing)

    show = commands.add_parser("show", help="print the tail of one metric matrix")
    _add_partition_flags(show)
    show.add_argument("metric")
    show.add_argument("--symbols", nargs="+")
    show.add_argument("--rows", type=int, default=10)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    partition = _partition(args)

    if args.command == "build":
        symbols = list(args.symbols) if args.symbols else []
        if args.symbols_file:
            symbols += [
                line.strip()
                for line in args.symbols_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        build(
            partition=partition,
            metrics_out=args.metrics_out,
            labels_out=args.labels_out,
            symbols=symbols or None,
            limit=args.limit,
            batch_size=args.batch_size,
            workers=args.workers,
            with_labels=not args.no_labels,
            with_cross_section=not args.no_cross_section,
            incremental=args.incremental,
            keep_staging=args.keep_staging,
        )
        return 0

    if args.command == "status":
        manifest = read_manifest(partition.metrics_dir)
        if manifest is None:
            print(f"no panel built for {partition}")
            return 1
        for key in ("partition", "created_utc", "rows", "symbols", "start", "end", "build_seconds"):
            print(f"{key:>14}: {manifest.get(key)}")
        print(f"{'metrics':>14}: {len(manifest.get('metrics', []))}")
        return 0

    if args.command == "list":
        names = available_metrics(partition.metrics_dir)
        if not names:
            print(f"no metrics under {partition.metrics_dir}")
            return 1
        for name in names:
            print(name)
        return 0

    if args.command == "show":
        path = matrix_path(partition.metrics_dir, args.metric)
        if not path.exists():
            path = matrix_path(partition.labels_dir, args.metric)
        if not path.exists():
            print(f"no such metric: {args.metric}")
            return 1
        frame = read_matrix(path, columns=args.symbols)
        print(frame.tail(args.rows).to_string())
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
