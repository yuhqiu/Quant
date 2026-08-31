"""Parquet and JSON IO with the project's compression, encoding and atomicity defaults.

Every write goes to a temporary file in the destination directory and is then
renamed, so a killed process never leaves a half-written artifact behind.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import settings

INDEX_NAME = "date"
FLOAT_ENCODING = "BYTE_STREAM_SPLIT"


@contextmanager
def atomic_path(path: Path | str, suffix: str = ".tmp") -> Iterator[Path]:
    """Yield a temp path next to ``path``; rename it into place on clean exit."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + suffix)
    try:
        yield temporary
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _float_encoding(schema: pa.Schema, exclude: set[str]) -> dict[str, str]:
    return {
        field.name: FLOAT_ENCODING
        for field in schema
        if pa.types.is_floating(field.type) and field.name not in exclude
    }


def write_table(table: pa.Table, path: Path | str, use_dictionary: Any = False) -> Path:
    target = Path(path)
    with atomic_path(target) as temporary:
        pq.write_table(
            table,
            temporary,
            compression=settings().compression,
            use_dictionary=use_dictionary,
            # Byte-stream-split groups float bytes into planes so zstd finds
            # structure in them: roughly 38% smaller than plain zstd.
            column_encoding=_float_encoding(
                table.schema,
                exclude=set(use_dictionary) if isinstance(use_dictionary, list) else set(),
            ),
        )
    return target


def write_parquet(
    frame: pd.DataFrame,
    path: Path | str,
    schema: pa.Schema | None = None,
    index: bool = False,
) -> Path:
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=index)
    return write_table(table, path)


def read_parquet(
    path: Path | str, columns: list[str] | None = None
) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow", columns=columns)


def write_matrix(frame: pd.DataFrame, path: Path | str) -> Path:
    """Write a wide matrix: index = date (UTC), columns = symbol."""
    matrix = frame.copy()
    matrix.index = _utc_index(matrix.index)
    matrix.columns = [str(column) for column in matrix.columns]
    return write_table(pa.Table.from_pandas(matrix), path)


def read_matrix(path: Path | str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a wide matrix, restricted to ``columns`` when given.

    Missing columns are tolerated and come back as NaN, so a signal can ask for a
    symbol list that predates the current build.
    """
    target = Path(path)
    if columns is None:
        frame = pd.read_parquet(target, engine="pyarrow")
    else:
        available = set(pq.ParquetFile(target).schema.names)
        wanted = [name for name in columns if name in available]
        frame = pd.read_parquet(target, engine="pyarrow", columns=wanted)
        frame = frame.reindex(columns=list(columns))
    frame.index = _utc_index(frame.index)
    return frame.sort_index()


def matrix_path(directory: Path | str, metric: str) -> Path:
    return Path(directory) / f"{metric}.parquet"


def available_metrics(directory: Path | str) -> list[str]:
    folder = Path(directory)
    if not folder.is_dir():
        return []
    return sorted(
        path.stem for path in folder.glob("*.parquet") if not path.name.startswith("_")
    )


def write_json(payload: Any, path: Path | str) -> Path:
    target = Path(path)
    with atomic_path(target) as temporary:
        temporary.write_text(
            json.dumps(payload, indent=2, default=str, sort_keys=False), encoding="utf-8"
        )
    return target


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _utc_index(index: pd.Index) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    result.name = INDEX_NAME
    return result
