"""Shared test harness.

Uses stdlib only, so it can be imported before ``QUANT_DATA_ROOT`` is set and the
``DataAcquisition`` package is first imported (``config`` resolves paths at
import time, so the env var has to be in place first).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def isolated_data_root(prefix: str) -> Path:
    """Put the project on sys.path and point the lake at a throwaway directory."""
    import os

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    root = Path(tempfile.mkdtemp(prefix=prefix))
    os.environ["QUANT_DATA_ROOT"] = str(root)
    return root


class Results:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []

    def check(self, name: str, condition: bool, detail: object = "") -> None:
        (self.passed if condition else self.failed).append(name)
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}{(' -> ' + str(detail)) if detail else ''}")

    def section(self, title: str) -> None:
        print(f"\n== {title} ==")

    def exit_code(self) -> int:
        print(f"\n{len(self.passed)} passed, {len(self.failed)} failed")
        if self.failed:
            print("FAILURES: " + ", ".join(self.failed))
        return 1 if self.failed else 0
