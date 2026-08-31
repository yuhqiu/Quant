"""Command line interface: ``python -m DataAcquisition <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import catalog, lake, migrate, quality, universe
from .config import (
    DEFAULT_ASSET_CLASS,
    DEFAULT_INTERVAL,
    DEFAULT_PROVIDER,
    DEFAULT_REGION,
    LAKE_ROOT,
)
from .ingest import ingest as run_ingest
from .lake import Partition
from .providers import provider_names


def _progress(quiet: bool):
    if quiet:
        return None

    def report(done: int, total: int, symbol: str, message: str) -> None:
        print(f"[{done}/{total}] {symbol}: {message}", flush=True)

    return report


def _partition(args: argparse.Namespace) -> Partition:
    return Partition(region=args.region, asset_class=args.asset_class, interval=args.interval)


def _add_partition_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--asset-class", default=DEFAULT_ASSET_CLASS, choices=["stock", "etf", "other"])
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)


def _print_frame(frame: pd.DataFrame, limit: int = 40) -> None:
    if frame.empty:
        print("(no rows)")
        return
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame.head(limit).to_string(index=False))
    if len(frame) > limit:
        print(f"... {len(frame) - limit} more rows")


def command_universe(args: argparse.Namespace) -> int:
    frame = universe.build_universe()
    path = universe.save_universe(frame)
    print(f"Universe snapshot written to {path}")
    print(frame["asset_class"].value_counts().to_string())
    return 0


def command_download(args: argparse.Namespace) -> int:
    report = run_ingest(
        symbols=args.symbols,
        provider=args.provider,
        region=args.region,
        asset_class=args.asset_class,
        interval=args.interval,
        start=args.start,
        end=args.end,
        mode=args.mode,
        symbols_file=args.symbols_file,
        use_universe=args.universe,
        refresh_universe=args.refresh_universe,
        batch_size=args.batch_size,
        pause=args.pause,
        drop_zero_volume=args.drop_zero_volume,
        on_progress=_progress(args.quiet),
    )
    print(report.summary())
    return 1 if report.failures and not report.updated else 0


def command_update(args: argparse.Namespace) -> int:
    """Incremental refresh: only bars newer than what the catalog already holds."""
    args.mode = "incremental" if args.strict else "auto"
    args.start = args.start or None
    return command_download(args)


def command_status(args: argparse.Namespace) -> int:
    _print_frame(catalog.status(args.interval))
    return 0


def command_quality(args: argparse.Namespace) -> int:
    partition = _partition(args)
    report = quality.build_report(partition)
    path = quality.save_report(report, partition)
    print(f"Quality report written to {path}")
    _print_frame(report, limit=args.limit)
    return 0


def command_migrate(args: argparse.Namespace) -> int:
    counts = migrate.migrate_csv_directory(
        source=args.source,
        partition=_partition(args),
        provider=args.provider,
        on_progress=_progress(args.quiet),
    )
    print(
        f"Migrated {counts['converted']} symbols ({counts['rows']} rows), "
        f"skipped {counts['skipped']}"
    )
    return 0


def command_refresh_state(args: argparse.Namespace) -> int:
    count = catalog.refresh_from_lake(provider=args.provider)
    print(f"Rebuilt catalog state for {count} symbol/interval entries from {LAKE_ROOT}")
    return 0


def command_query(args: argparse.Namespace) -> int:
    frame = catalog.query(args.sql)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.output, engine="pyarrow", compression="zstd", index=False)
        print(f"{len(frame)} rows written to {args.output}")
    else:
        _print_frame(frame, limit=args.limit)
    return 0


def command_export(args: argparse.Namespace) -> int:
    partition = _partition(args)
    frame = lake.read_bars(partition, symbols=args.symbols)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".csv":
        frame.to_csv(destination, index=False)
    else:
        frame.to_parquet(destination, engine="pyarrow", compression="zstd", index=False)
    print(f"{len(frame)} rows written to {destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m DataAcquisition",
        description="Download, store and audit market data in a parquet lake queried by DuckDB.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    universe_parser = subparsers.add_parser("universe", help="Refresh the US listing snapshot.")
    universe_parser.set_defaults(handler=command_universe)

    for name, handler, help_text in (
        ("download", command_download, "Download bars into the lake."),
        ("update", command_update, "Download only bars newer than the stored watermark."),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--symbols", "--symbol", nargs="+", dest="symbols")
        sub.add_argument("--symbols-file", help="File with whitespace or comma separated symbols.")
        sub.add_argument("--universe", action="store_true", help="Use the whole asset-class universe.")
        sub.add_argument("--refresh-universe", action="store_true", help="Re-download the universe first.")
        _add_partition_arguments(sub)
        sub.add_argument("--provider", default=DEFAULT_PROVIDER, choices=provider_names())
        sub.add_argument("--start", help="Inclusive UTC start date, e.g. 2006-01-01.")
        sub.add_argument("--end", help="Exclusive UTC end date.")
        sub.add_argument("--batch-size", type=int)
        sub.add_argument("--pause", type=float, help="Seconds between provider requests.")
        sub.add_argument("--drop-zero-volume", action="store_true")
        sub.add_argument("--quiet", action="store_true")
        if name == "download":
            sub.add_argument("--mode", default="auto", choices=["auto", "full", "incremental"])
        else:
            sub.add_argument("--strict", action="store_true", help="Fail if no watermark exists.")
        sub.set_defaults(handler=handler)

    status_parser = subparsers.add_parser("status", help="Show stored coverage per partition.")
    status_parser.add_argument("--interval")
    status_parser.set_defaults(handler=command_status)

    quality_parser = subparsers.add_parser("quality", help="Build the per-symbol quality report.")
    _add_partition_arguments(quality_parser)
    quality_parser.add_argument("--limit", type=int, default=20)
    quality_parser.set_defaults(handler=command_quality)

    migrate_parser = subparsers.add_parser("migrate", help="Import legacy CSV files into the lake.")
    migrate_parser.add_argument("source", help="Directory holding the legacy CSV files.")
    _add_partition_arguments(migrate_parser)
    migrate_parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    migrate_parser.add_argument("--quiet", action="store_true")
    migrate_parser.set_defaults(handler=command_migrate)

    refresh_parser = subparsers.add_parser(
        "refresh-state", help="Rebuild catalog watermarks by scanning the lake."
    )
    refresh_parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    refresh_parser.set_defaults(handler=command_refresh_state)

    query_parser = subparsers.add_parser("query", help="Run SQL against the lake.")
    query_parser.add_argument("sql")
    query_parser.add_argument("--output", help="Write the result to a parquet file instead.")
    query_parser.add_argument("--limit", type=int, default=40)
    query_parser.set_defaults(handler=command_query)

    export_parser = subparsers.add_parser("export", help="Export a partition to parquet or CSV.")
    _add_partition_arguments(export_parser)
    export_parser.add_argument("--symbols", nargs="+")
    export_parser.add_argument("--output", required=True)
    export_parser.set_defaults(handler=command_export)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
