"""Build wide metric matrices (date x symbol) from the parquet bar lake.

Output is one Parquet file per metric, indexed by date with one column per symbol::

    close = read_matrix("Metrics/US/stock/1d/close.parquet")
    rsi = read_matrix("Metrics/US/stock/1d/rsi_14.parquet")

Wide Parquet cannot be appended to, so an update is a full rebuild. Incremental
mode therefore means "skip the rebuild when the lake has nothing new", which is
what makes a daily pipeline cheap and this stage idempotent.
"""

from __future__ import annotations

import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Common.io import matrix_path, read_json, read_matrix, write_json, write_matrix
from Common.logging import get_logger
from Common.provenance import stamp
from Common.types import Partition
from DataAcquisition import stored_symbols
from MetricsGeneration import cross_section
from MetricsGeneration.indicators import compute_labels, compute_metrics, metric_dtype
from MetricsGeneration.storage import load_bars

STAGING_DIRNAME = "_staging"
MANIFEST_FILENAME = "_manifest.json"
MIN_ROWS = 2

log = get_logger(__name__)


def resolve_symbols(
    partition: Partition,
    symbols: list[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Symbols to build, restricted to those actually present in the lake."""
    available = stored_symbols(partition)
    if not available:
        raise ValueError(f"no bars stored for partition {partition}")

    if symbols:
        wanted = {symbol.strip().upper() for symbol in symbols}
        available = [symbol for symbol in available if symbol.upper() in wanted]

    available.sort()
    if limit is not None:
        available = available[:limit]
    if not available:
        raise ValueError(f"no matching symbols in partition {partition}")
    return available


def lake_end_date(partition: Partition, symbols: list[str]) -> str | None:
    """Newest bar date in the partition, read from parquet statistics only."""
    from DataAcquisition.lake import last_timestamp

    newest: pd.Timestamp | None = None
    for symbol in symbols:
        candidate = last_timestamp(symbol, partition)
        if candidate is not None and (newest is None or candidate > newest):
            newest = candidate
    return None if newest is None else newest.date().isoformat()


def read_manifest(metrics_out: Path | str) -> dict | None:
    path = Path(metrics_out) / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (ValueError, OSError):
        return None


def _write_shards(frames: dict[str, pd.DataFrame], root: Path, batch_id: int) -> None:
    """Stage one parquet file per metric for this batch of symbols.

    Assembled a metric at a time rather than by concatenating every symbol's full
    frame at once: the wide intermediate is the difference between 15 MB and a
    gigabyte per worker.
    """
    if not frames:
        return

    symbols = sorted(frames)
    index = frames[symbols[0]].index
    for symbol in symbols[1:]:
        index = index.union(frames[symbol].index)
    index = index.sort_values()

    for metric in frames[symbols[0]].columns:
        dtype = metric_dtype(metric)
        matrix = pd.DataFrame(
            {
                symbol: frames[symbol][metric].reindex(index).to_numpy(dtype=dtype)
                for symbol in symbols
            },
            index=index,
        )
        write_matrix(matrix, root / metric / f"part-{batch_id:05d}.parquet")


def process_batch(
    symbols: list[str],
    partition: Partition,
    staging_root: Path,
    batch_id: int,
    with_labels: bool,
) -> tuple[int, list[str]]:
    """Compute one batch of symbols and stage the results as per-metric shards."""
    metric_frames: dict[str, pd.DataFrame] = {}
    label_frames: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for symbol in symbols:
        try:
            bars = load_bars(symbol, partition)
            if len(bars) < MIN_ROWS:
                failed.append(symbol)
                continue
            metric_frames[symbol] = compute_metrics(bars)
            if with_labels:
                label_frames[symbol] = compute_labels(bars)
        except Exception:
            failed.append(symbol)

    _write_shards(metric_frames, staging_root / "metrics", batch_id)
    _write_shards(label_frames, staging_root / "labels", batch_id)
    return len(metric_frames), failed


def assemble_metric(staging_root: Path, out_dir: Path, metric: str) -> tuple[int, int]:
    """Concatenate every shard of one metric along the symbol axis into a final matrix."""
    shards = sorted(Path(staging_root, metric).glob("part-*.parquet"))
    matrix = pd.concat([read_matrix(shard) for shard in shards], axis=1).sort_index()
    matrix = matrix.reindex(columns=sorted(matrix.columns))
    write_matrix(matrix, matrix_path(out_dir, metric))
    return matrix.shape


def _staged_metric_names(staging_root: Path) -> list[str]:
    if not staging_root.exists():
        return []
    return sorted(path.name for path in staging_root.iterdir() if path.is_dir())


def _write_manifest(
    out_dir: Path,
    partition: Partition,
    symbols: list[str],
    metrics: list[str],
    elapsed: float,
) -> None:
    reference = read_matrix(matrix_path(out_dir, "close"))
    manifest = stamp(
        source=str(partition.path),
        partition=partition.key,
        layout="wide: index=date (UTC), columns=symbol, one parquet file per metric",
        rows=int(reference.shape[0]),
        symbols=int(reference.shape[1]),
        start=reference.index.min().date().isoformat(),
        end=reference.index.max().date().isoformat(),
        build_seconds=round(elapsed, 1),
        requested_symbols=len(symbols),
        metrics=metrics,
        symbol_list=list(reference.columns),
    )
    write_json(manifest, Path(out_dir) / MANIFEST_FILENAME)


def build(
    partition: Partition | None = None,
    metrics_out: Path | str | None = None,
    labels_out: Path | str | None = None,
    symbols: list[str] | None = None,
    limit: int | None = None,
    batch_size: int = 100,
    workers: int = 8,
    with_labels: bool = True,
    with_cross_section: bool = True,
    incremental: bool = False,
    keep_staging: bool = False,
    verbose: bool = True,
) -> Path:
    """Rebuild the feature panel for one partition. Returns the metrics directory."""
    partition = partition or Partition()
    metrics_out = Path(metrics_out) if metrics_out else partition.metrics_dir
    labels_out = Path(labels_out) if labels_out else partition.labels_dir

    started = time.perf_counter()
    names = resolve_symbols(partition, symbols, limit)

    if incremental:
        manifest = read_manifest(metrics_out)
        latest = lake_end_date(partition, names)
        if manifest and latest and str(manifest.get("end", "")) >= latest:
            log.info("metrics up to date", extra={"partition": partition.key, "end": latest})
            if verbose:
                print(f"up to date at {latest} -> {metrics_out}")
            return metrics_out

    batches = [names[i : i + batch_size] for i in range(0, len(names), batch_size)]
    staging_root = metrics_out / STAGING_DIRNAME
    if staging_root.exists():
        shutil.rmtree(staging_root)

    if verbose:
        print(f"[1/3] computing metrics for {len(names)} symbols in {len(batches)} batches")
    processed = 0
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_batch, batch, partition, staging_root, index, with_labels
            ): index
            for index, batch in enumerate(batches)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            count, failed = future.result()
            processed += count
            failures.extend(failed)
            if verbose:
                print(f"      batch {done}/{len(batches)} -> {processed} symbols", end="\r")

    if verbose:
        print(f"\n      {processed} symbols computed, {len(failures)} skipped")
        if failures:
            print(f"      skipped: {', '.join(sorted(failures)[:20])}")
    log.info(
        "per-symbol metrics complete",
        extra={"computed": processed, "skipped": len(failures)},
    )

    metric_names = _staged_metric_names(staging_root / "metrics")
    label_names = _staged_metric_names(staging_root / "labels")
    if not metric_names:
        raise ValueError("no metrics were staged; every symbol failed")

    if verbose:
        print(f"[2/3] assembling {len(metric_names) + len(label_names)} matrices")
    jobs = [(staging_root / "metrics", metrics_out, name) for name in metric_names]
    jobs += [(staging_root / "labels", labels_out, name) for name in label_names]
    with ProcessPoolExecutor(max_workers=max(1, min(4, workers))) as executor:
        futures = {executor.submit(assemble_metric, *job): job[2] for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            rows, columns = future.result()
            if verbose:
                print(f"      {done}/{len(jobs)} {futures[future]} ({rows} x {columns})", end="\r")
    if verbose:
        print()

    if with_cross_section:
        if verbose:
            print("[3/3] cross-sectional and market-relative metrics")
        metric_names += cross_section.run(metrics_out, verbose=verbose)
    elif verbose:
        print("[3/3] skipped cross-sectional stage")

    if not keep_staging:
        shutil.rmtree(staging_root, ignore_errors=True)

    elapsed = time.perf_counter() - started
    _write_manifest(metrics_out, partition, names, sorted(set(metric_names)), elapsed)
    log.info(
        "metrics build complete",
        extra={"partition": partition.key, "seconds": round(elapsed, 1)},
    )
    if verbose:
        print(f"done in {elapsed:.1f}s -> {metrics_out}")
    return metrics_out


def main(argv: list[str] | None = None) -> int:
    from MetricsGeneration.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
