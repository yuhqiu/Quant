"""Build wide metric matrices (date x symbol) from cleaned daily OHLCV CSVs.

Output is one Parquet file per metric, indexed by date with one column per symbol,
which is the shape vectorbt consumes directly:

    close = pd.read_parquet("Metrics/US/Stock/day/close.parquet")
    rsi = pd.read_parquet("Metrics/US/Stock/day/rsi_14.parquet")
    pf = vbt.Portfolio.from_signals(close, rsi < 30, rsi > 70)

Wide Parquet cannot be appended to, so updates are full rebuilds; a rebuild is
cheap enough that this is simpler and safer than merging partitions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MetricsGeneration import cross_section
from MetricsGeneration.indicators import compute_labels, compute_metrics, metric_dtype
from MetricsGeneration.storage import (
    load_ohlcv,
    matrix_path,
    read_matrix,
    write_matrix,
)

DEFAULT_SOURCE = PROJECT_ROOT / "DataSource" / "US" / "Stock" / "day"
DEFAULT_METRICS_OUT = PROJECT_ROOT / "Metrics" / "US" / "Stock" / "day"
DEFAULT_LABELS_OUT = PROJECT_ROOT / "Metrics" / "US" / "Stock" / "labels_day"

STAGING_DIRNAME = "_staging"
MANIFEST_FILENAME = "_manifest.json"
MIN_ROWS = 2


def discover_symbol_files(
    source: Path,
    symbols: list[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    if symbols:
        wanted = {symbol.upper() for symbol in symbols}
        files = [path for path in source.glob("*.csv") if path.stem.upper() in wanted]
    else:
        files = list(source.glob("*.csv"))

    files.sort(key=lambda path: path.stem)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError(f"no matching CSV files under {source}")
    return files


def _write_shards(frames: dict[str, pd.DataFrame], root: Path, batch_id: int) -> None:
    if not frames:
        return
    combined = pd.concat(frames, axis=1).sort_index()
    for metric in next(iter(frames.values())).columns:
        matrix = combined.xs(metric, axis=1, level=1).astype(metric_dtype(metric))
        write_matrix(matrix, root / metric / f"part-{batch_id:05d}.parquet")


def process_batch(
    paths: list[Path],
    staging_root: Path,
    batch_id: int,
    with_labels: bool,
) -> tuple[int, list[str]]:
    """Compute one batch of symbols and stage the results as per-metric shards."""
    metric_frames: dict[str, pd.DataFrame] = {}
    label_frames: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for path in paths:
        symbol = path.stem
        try:
            ohlcv = load_ohlcv(path)
            if len(ohlcv) < MIN_ROWS:
                failed.append(symbol)
                continue
            metric_frames[symbol] = compute_metrics(ohlcv)
            if with_labels:
                label_frames[symbol] = compute_labels(ohlcv)
        except Exception:
            failed.append(symbol)

    _write_shards(metric_frames, staging_root / "metrics", batch_id)
    _write_shards(label_frames, staging_root / "labels", batch_id)
    return len(metric_frames), failed


def assemble_metric(staging_root: Path, out_dir: Path, metric: str) -> tuple[int, int]:
    """Concatenate every shard of one metric along the symbol axis into a final matrix."""
    shards = sorted((staging_root / metric).glob("part-*.parquet"))
    matrix = pd.concat([read_matrix(shard) for shard in shards], axis=1).sort_index()
    matrix = matrix.reindex(columns=sorted(matrix.columns))
    write_matrix(matrix, matrix_path(out_dir, metric))
    return matrix.shape


def _staged_metric_names(staging_root: Path) -> list[str]:
    if not staging_root.exists():
        return []
    return sorted(path.name for path in staging_root.iterdir() if path.is_dir())


def _write_manifest(out_dir: Path, source: Path, metrics: list[str], elapsed: float) -> None:
    reference = read_matrix(matrix_path(out_dir, "close"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source),
        "layout": "wide: index=date (UTC), columns=symbol, one parquet file per metric",
        "rows": int(reference.shape[0]),
        "symbols": int(reference.shape[1]),
        "start": reference.index.min().date().isoformat(),
        "end": reference.index.max().date().isoformat(),
        "build_seconds": round(elapsed, 1),
        "pandas": pd.__version__,
        "metrics": metrics,
        "symbol_list": list(reference.columns),
    }
    (out_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build(
    source: Path,
    metrics_out: Path,
    labels_out: Path,
    symbols: list[str] | None = None,
    limit: int | None = None,
    batch_size: int = 100,
    workers: int = 8,
    with_labels: bool = True,
    with_cross_section: bool = True,
    keep_staging: bool = False,
) -> None:
    started = time.perf_counter()
    files = discover_symbol_files(source, symbols, limit)
    batches = [files[start : start + batch_size] for start in range(0, len(files), batch_size)]

    staging_root = metrics_out / STAGING_DIRNAME
    if staging_root.exists():
        shutil.rmtree(staging_root)

    print(f"[1/3] computing metrics for {len(files)} symbols in {len(batches)} batches")
    processed = 0
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_batch, batch, staging_root, index, with_labels): index
            for index, batch in enumerate(batches)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            count, failed = future.result()
            processed += count
            failures.extend(failed)
            print(f"      batch {done}/{len(batches)} -> {processed} symbols", end="\r")

    print(f"\n      {processed} symbols computed, {len(failures)} skipped")
    if failures:
        print(f"      skipped: {', '.join(sorted(failures)[:20])}")

    metric_names = _staged_metric_names(staging_root / "metrics")
    label_names = _staged_metric_names(staging_root / "labels")

    print(f"[2/3] assembling {len(metric_names) + len(label_names)} matrices")
    assembly_workers = max(1, min(4, workers))
    jobs = [(staging_root / "metrics", metrics_out, name) for name in metric_names]
    jobs += [(staging_root / "labels", labels_out, name) for name in label_names]
    with ProcessPoolExecutor(max_workers=assembly_workers) as executor:
        futures = {
            executor.submit(assemble_metric, *job): job[2] for job in jobs
        }
        for done, future in enumerate(as_completed(futures), start=1):
            rows, columns = future.result()
            print(f"      {done}/{len(jobs)} {futures[future]} ({rows} x {columns})", end="\r")
    print()

    if with_cross_section:
        print("[3/3] cross-sectional and market-relative metrics")
        metric_names += cross_section.run(metrics_out)
    else:
        print("[3/3] skipped cross-sectional stage")

    if not keep_staging:
        shutil.rmtree(staging_root, ignore_errors=True)

    elapsed = time.perf_counter() - started
    _write_manifest(metrics_out, source, sorted(metric_names), elapsed)
    print(f"done in {elapsed:.1f}s -> {metrics_out}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    parser.add_argument("--labels-out", type=Path, default=DEFAULT_LABELS_OUT)
    parser.add_argument("--symbols", nargs="+", help="restrict the build to these tickers")
    parser.add_argument("--limit", type=int, help="use only the first N symbols")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--no-cross-section", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build(
        source=args.source,
        metrics_out=args.metrics_out,
        labels_out=args.labels_out,
        symbols=args.symbols,
        limit=args.limit,
        batch_size=args.batch_size,
        workers=args.workers,
        with_labels=not args.no_labels,
        with_cross_section=not args.no_cross_section,
        keep_staging=args.keep_staging,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
