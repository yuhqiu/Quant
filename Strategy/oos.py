"""Out-of-sample discipline.

The held-back window is a one-shot resource. Every time a configuration touches
it the fact is written down, because a Sharpe from the fortieth look at the test
set is not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from Common.config import settings
from Common.io import read_json, write_json
from Common.logging import get_logger

LEDGER_FILENAME = "oos_ledger.json"
log = get_logger(__name__)


@dataclass(frozen=True)
class OosRecord:
    spec_hash: str
    strategy: str
    touches: int
    first_seen: str
    last_seen: str
    windows: list[str]


def ledger_path(root: Path | str | None = None) -> Path:
    base = Path(root) if root else settings().results_root
    return base / LEDGER_FILENAME


def read_ledger(root: Path | str | None = None) -> dict[str, dict]:
    path = ledger_path(root)
    if not path.is_file():
        return {}
    try:
        return dict(read_json(path))
    except (ValueError, OSError):
        return {}


def record_touch(
    spec_hash: str,
    strategy: str,
    window: str,
    root: Path | str | None = None,
) -> OosRecord:
    """Register one evaluation against the out-of-sample window."""
    ledger = read_ledger(root)
    now = datetime.now(UTC).isoformat(timespec="seconds")

    entry = ledger.get(spec_hash) or {
        "spec_hash": spec_hash,
        "strategy": strategy,
        "touches": 0,
        "first_seen": now,
        "windows": [],
    }
    entry["touches"] = int(entry["touches"]) + 1
    entry["last_seen"] = now
    entry["strategy"] = strategy
    if window not in entry["windows"]:
        entry["windows"].append(window)

    ledger[spec_hash] = entry
    write_json(ledger, ledger_path(root))

    record = OosRecord(
        spec_hash=spec_hash,
        strategy=strategy,
        touches=int(entry["touches"]),
        first_seen=str(entry["first_seen"]),
        last_seen=str(entry["last_seen"]),
        windows=list(entry["windows"]),
    )
    if record.touches > 1:
        log.warning(
            "out-of-sample window touched more than once",
            extra={"spec_hash": spec_hash[:12], "touches": record.touches},
        )
    return record


def touches(spec_hash: str, root: Path | str | None = None) -> int:
    entry = read_ledger(root).get(spec_hash)
    return int(entry["touches"]) if entry else 0
