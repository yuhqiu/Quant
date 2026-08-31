"""Project logging: JSON lines to disk for machines, plain text to console for humans."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from .config import settings

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_RESERVED = frozenset(vars(logging.makeLogRecord({})))
_configured = False


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with any extra=... fields merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in vars(record).items()
            if key not in _RESERVED and not key.startswith("_")
        }
        payload.update(extras)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str | int | None = None,
    log_dir: Path | None = None,
    console: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Install handlers on the root logger. Idempotent unless ``force``."""
    global _configured
    root = logging.getLogger("quant")
    if _configured and not force:
        return root

    resolved = settings()
    root.setLevel(level or resolved.log_level)
    root.handlers.clear()
    root.propagate = False

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(stream)

    if resolved.log_json:
        directory = Path(log_dir) if log_dir else resolved.log_dir
        try:
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%d")
            handler = logging.FileHandler(
                directory / f"quant-{stamp}.jsonl", encoding="utf-8"
            )
            handler.setFormatter(JsonFormatter())
            root.addHandler(handler)
        except OSError:
            # A read-only or missing lake must not stop the program from running.
            pass

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Logger under the ``quant`` namespace, configuring handlers on first use."""
    configure_logging()
    suffix = name.removeprefix("quant.")
    return logging.getLogger(f"quant.{suffix}")
