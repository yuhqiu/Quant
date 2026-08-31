"""Walk-forward splitting with purge and embargo.

Overlapping labels leak: a 21-day forward return known at the end of the training
window still describes days that fall inside the test window. The purge removes
that overlap and the embargo adds a gap on the other side.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def label(self) -> str:
        return f"{self.test_start.date()}_{self.test_end.date()}"

    def as_dict(self) -> dict[str, str]:
        return {
            "train_start": self.train_start.date().isoformat(),
            "train_end": self.train_end.date().isoformat(),
            "test_start": self.test_start.date().isoformat(),
            "test_end": self.test_end.date().isoformat(),
        }


def walk_forward(
    dates: pd.DatetimeIndex,
    train_size: int = 756,
    test_size: int = 252,
    step: int | None = None,
    purge: int = 21,
    embargo: int = 5,
    anchored: bool = False,
) -> list[Split]:
    """Rolling (or anchored) train/test windows separated by ``purge + embargo`` bars."""
    index = pd.DatetimeIndex(dates).sort_values()
    total = len(index)
    step = step or test_size
    gap = purge + embargo

    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")

    splits: list[Split] = []
    start = 0
    while True:
        train_stop = start + train_size
        test_start = train_stop + gap
        test_stop = test_start + test_size
        if test_start >= total:
            break
        test_stop = min(test_stop, total)

        splits.append(
            Split(
                train_start=index[0 if anchored else start],
                train_end=index[train_stop - 1],
                test_start=index[test_start],
                test_end=index[test_stop - 1],
            )
        )
        if test_stop >= total:
            break
        start += step
    return splits


def in_sample_split(
    dates: pd.DatetimeIndex, fraction: float = 0.7
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Boundary between the development window and the held-back window."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    index = pd.DatetimeIndex(dates).sort_values()
    cut = int(len(index) * fraction)
    return index[cut - 1], index[cut]
