"""Test support for DataAcquisition."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def isolated_data_root(prefix: str) -> Iterator[Path]:
    """Point the lake at a throwaway directory, and put it back afterwards.

    Every path in the project resolves through ``Common.config.settings()`` on
    access, so redirecting the environment and clearing the cache is enough.
    """
    from Common import config

    previous = os.environ.get("QUANT_DATA_ROOT")
    root = Path(tempfile.mkdtemp(prefix=prefix))
    os.environ["QUANT_DATA_ROOT"] = str(root)
    config.reset_settings()
    try:
        yield root
    finally:
        if previous is None:
            os.environ.pop("QUANT_DATA_ROOT", None)
        else:
            os.environ["QUANT_DATA_ROOT"] = previous
        config.reset_settings()
        shutil.rmtree(root, ignore_errors=True)


def live_tests_enabled() -> bool:
    return os.environ.get("QUANT_LIVE_TESTS", "") == "1"
